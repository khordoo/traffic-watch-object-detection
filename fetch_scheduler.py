import requests
import psycopg2
from datetime import datetime
import time
from collections import defaultdict
import json
from dateutil import tz

class Postgres:
    def __init__(self):
        self.conn = psycopg2.connect(host="localhost", database="azure_ai", user="postgres", password="postgres")
        self.cursor = self.conn.cursor()
        self.cursor.execute('DELETE FROM camera')
        self.conn.commit()

    def insert_camera(self, camera):
        self.cursor.execute(
            f"INSERT INTO camera (id,address,region,latitude,longitude) VALUES({camera['id']}, '{camera['address']}', '{camera['region']}', {camera['latitude']}, {camera['longitude']})")

        self.conn.commit()
        print('Inserted camera:', camera['id'])


    def insert(self, query):
        self.cursor.execute(query)
        self.conn.commit()


class Scheduler:
    def __init__(self):
        self.database = Postgres()
        self.cameras = self.get_camera_locations()
        self.insert_cameras(self.cameras)
        self.session = requests.Session()
        self.detection_url = 'http://127.0.0.1:5000/analysis'
        # self.fetch_images()
        self.FETCHING_INTERVAL_SEC = 300

    def insert_cameras(self, cameras):
        for camera in cameras:
            self.database.insert_camera(camera)

    def insert_detection(self, detection):
        # insert into counts
        confidences = detection['countsConfidence']
        # Insert counts
        for object_type, count in detection['counts'].items():
            confidence = confidences[object_type]
            counts = f"INSERT INTO count (camera_id , time,label,count, confidence) VALUES({detection['camera']['id']},'{detection['time']}','{object_type}',{count},{confidence})"
            self.database.insert(counts)

        # Insert indivdual objects
        for item in detection["detection"]:
            x0 = item['topleft']['x']
            y0 = item['topleft']['y']
            x1 = item['bottomright']['x']
            y1 = item['bottomright']['y']
            xc = x0 + 0.5 * (x1 - x0)
            yc = y0 + 0.5 * (y1 - y0)
            query = f"INSERT INTO object_location (camera_id ,time  ,label  , x_top_left ,y_top_left  ,x_bottom_right , y_bottom_right ,x_center , y_center ,confidence  ) " \
                    f"VALUES ({detection['camera']['id']},'{detection['time']}' , '{item['label']}',{x0},{y0} ,{x1} ,{y1} ,{xc},{yc},{item['confidence']} )"
            self.database.insert(query)
            print('Inserte object: ', item['label'])

    def get_camera_locations(self):
        cameras = []
        response = requests.get('https://data.calgary.ca/api/views/6fv8-ymsc/rows.json')
        locations = response.json()['data']
        for location in locations:
            cameras.append({
                'address': location[8],
                'image_url': location[10][0],
                'id': int(location[10][0].split('/')[-1].split('.')[0][3:]),
                'region': location[9],
                'latitude': float(location[12]),
                'longitude': float(location[11]),
            })
        return cameras

    def run(self):
        while True:
            try:
                start_time = datetime.now()
                i = 0
                for detection in self.fetch_images():
                    i += 1
                    print(f'Insert detection: {i}')
                    self.insert_detection(detection)

                elapsed_time = (datetime.now() - start_time).total_seconds()
                waitting_time = self.FETCHING_INTERVAL_SEC - elapsed_time
                print('Elapsed time: ', elapsed_time)
                print('Waitting time:', waitting_time)
                if waitting_time < 0 or elapsed_time > self.FETCHING_INTERVAL_SEC:
                    waitting_time = 0
                print('Waiting for :', waitting_time, ' sec.')
                time.sleep(waitting_time)  # 5 minutes
            except Exception as err:
                print(f'Exception happened :{err} , Time : ', datetime.now())

    def fetch_images(self):
        detections = []
        i = 0
        for camera in self.cameras:
            counts=defaultdict(int)
            confidence=defaultdict(float)
            i += 1
            print(f'Fetching camera {i}/{len(self.cameras)}')
            detection = self.session.post(self.detection_url, json={
                "image": camera['image_url']
            })
            detection = detection.json()['detections']
            detection=json.loads(detection)
            for item in detection:
                counts[item['label']] += 1
                confidence[item['label']] += float(item['confidence'])
            # Average confidence per group
            for key, value in confidence.items():
                confidence[key] = value / counts[key]
            utc_time=datetime.now(tz=tz.UTC)
            calgary_time = utc_time.astimezone(tz.gettz('America/Edmonton')).isoformat()
            #print('Calgry time:',calgary_time) 
            detections.append(
                {"camera": camera, "detection": detection, "counts": counts, "countsConfidence": confidence,
                 "time": calgary_time})

        return detections


Scheduler().run()
