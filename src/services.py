import cv2
import base64
import skimage.io as io
import numpy as np
import psycopg2
from werkzeug.exceptions import BadRequest
from collections import defaultdict
from ast import literal_eval as make_tuple
from src.yolo import YOLO


class ImageAnalysisService:
    """This is a image detection class that identifies the location and name of objects
       in an image.It uses Yolo for object detection task. If requested, It  also returns the original image with
       overlaid object bounding boxes and  their labels.
    """

    def __init__(self):
        self.yolo = YOLO()
        self.CONFIDENCE_THRESHOLD = 0.3
        self.DEFAULT_DRAW_COLOR = (0, 255, 0)
        self.target_labels = ['car', 'person', 'bus', 'truck', 'bicycle', 'motorbike']

    def detect(self, payload):
        """Detects the object in image using YOLO"""
        return self._run_detection(self._parse_params(payload))

    def _parse_params(self, payload):
        """Parses the request params"""
        try:
            default_params = {}
            default_params['image'] = payload['image']
            default_params['createImage'] = payload.get('createImage', False)
            default_params['summerize'] = payload.get('summerize', False)
            default_params['confidenceThreshold'] = payload.get('confidenceThreshold', self.CONFIDENCE_THRESHOLD)
            if 'drawRGBColor' in payload:
                default_params['drawRGBColor'] = make_tuple(payload['drawRGBColor'])
            else:
                default_params['drawRGBColor'] = self.DEFAULT_DRAW_COLOR

            return default_params

        except Exception as err:
            raise BadRequest(f"Bad Request: {err}")

    def _run_detection(self, params):
        image = self._parse_image(params)
        detections = self.yolo.detect(image)
        return self._format_results(image, detections,params)

    def _parse_image(self, params):
        if isinstance(params['image'], str) and params['image'].startswith('http'):
            return io.imread(params['image'])
        else:
            return params['image']

    def _format_results(self, image, detections, params):

        if params['createImage']:
            image = self._draw_objects(image, detections, params)
            image = self.base64_encoded(image)
        else:
            image = None

        if params['summerize']:
            detections = self._group_detections(detections)
        else:
            for detection in detections:
                detection['confidence'] = str(detection['confidence'])

        return {'detections': detections, 'image': image}

    def _group_detections(self, detections):
        groups = defaultdict(int)
        for detection in detections:
            if detection['label'] in self.target_labels:
                groups[detection['label']] += 1
        return groups

    def _draw_objects(self, image, detections, params):

        for detection in detections:
            if detection['label'] in self.target_labels and detection['confidence'] >= params['confidenceThreshold']:
                point_top_left = (detection['topleft']['x'], detection['topleft']['y'])
                point_bottom_right = (detection['bottomright']['x'], detection['bottomright']['y'])
                image = cv2.rectangle(image, point_top_left, point_bottom_right, params['drawRGBColor'], 2)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image_resize(image, height=200, width=400)
        return image

    def base64_encoded(self, image):
        _, values = cv2.imencode('.jpg', image)
        return base64.b64encode(values)


class Database:
    """Fetches the historical detection results from Postgres database"""

    def __init__(self, config):
        try:
            conn = psycopg2.connect(host=config.postgres_host, database=config.postgres_database_name,
                                    user=config.postgres_username, password=config.postgres_password)
            self.cursor = conn.cursor()
        except Exception as err:
            print(f'Failed to connect to Postgres database: {err}')

    def fetch(self, query):
        self.cursor.execute(query)
        detection_names = [desc[0] for desc in self.cursor.description]
        detection_values = self.cursor.fetchall()
        grouped = [dict(zip(detection_names, detection))
                   for detection in detection_values
                   ]
        return grouped

    def fetch_counts(self, camera_id=75, entity='car', time_bucket='hour', number_of_records=24):
        """Fetches the  top {} most recent records for the {entity} from the database"""
        query = f"select date_trunc('{time_bucket}',time) as ttime,camera_id,label, avg(count) from public.count group by date_trunc('{time_bucket}',time) ,camera_id,label having camera_id={camera_id} and label ='{entity}'  order by ttime desc limit {number_of_records};"
        self.cursor.execute(query)
        return [[record[0], float(record[-1])] for record in self.cursor.fetchall()]


class HistoricalImagePoinst:
    """Draws the historical location of detected objects on the Image."""

    def __init__(self, database):
        self.database = database
        self.image_url = 'http://trafficcam.calgary.ca/loc{}.jpg'
        self.DEFAULT_DRAW_COLOR = (0, 255, 0)
        self.CONFIDENCE_THRESHOLD = 0.3

    def draw(self, request):
        """Draws a circle at the central detected location of an object """
        params = self._parse_params(request)
        query = f"select * from public.object_location where camera_id={params['cameraId']} and label='{params['label']}' and confidence >{params['confidenceThreshold']} ;"
        records = self.database.fetch(query)
        image = io.imread(self.image_url.format(params['cameraId']))

        for record in records:
            color = self._apply_oppacity(params['color'], record['confidence'])
            image = cv2.circle(image, (int(record['x_center']), int(record['y_center'])), 5, color, -1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.base64_encoded(image_rgb)

    def _parse_params(self, request):
        """Parses the request params"""
        try:
            default_params = {}
            payload = request.get_json(force=True)
            default_params['cameraId'] = payload['cameraId']
            default_params['label'] = payload.get('label', 'car')
            default_params['confidenceThreshold'] = payload.get('confidenceThreshold', self.CONFIDENCE_THRESHOLD)
            if 'color' in payload:
                default_params['color'] = make_tuple(payload['color'])
            else:
                default_params['color'] = self.DEFAULT_DRAW_COLOR

            return default_params

        except Exception as err:
            raise BadRequest(f"Bad Request: {err}")

    def _apply_oppacity(self, color, opacity):
        if opacity > 1:
            opacity = 0.99
        color = np.array(color)
        color = color * float(opacity)
        return tuple(map(int, color))

    def base64_encoded(self, image):
        _, values = cv2.imencode('.jpg', image)
        return base64.b64encode(values)


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized
