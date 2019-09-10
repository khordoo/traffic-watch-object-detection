

import cv2
import numpy as np
from yolo import YOLO
from flask import Flask
import pandas as pd
from flask import request
import json
from PIL import Image
from io import BytesIO
import skimage.io as io
import base64



app = Flask(__name__)
camera = YOLO()

@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  return response

@app.route("/" ,methods = ['POST' ,'GET'] )
def welcome():
    return """Welcome """

@app.route("/v1/detection" ,methods = ['GET'] )
def index():
    return """It works! \n Use POST request for image analysis."""

@app.route('/v1/detection', methods = ['POST'])
def detect_objects():
    try:
       request_params = request.get_json(force=True)
    except:
       return "Bad Request.Request json string is not properly formatted"

    if 'output' in request_params:
        output_format = request_params['output']
    else:
        output_format="json"

    if 'minify' in request_params:
        output_minify=request_params['minify']
    else:
        output_minify = False

    if "distance" in request_params:
        distance=request_params['distance']
    else:
        distance = 'none'

    input_image_format = get_image_format(request_params)


    if input_image_format =='url':
        image_url = request_params['image']
        detected_objets = camera.detect(image_url, input_image_format)
        RGB_image=io.imread(image_url)
    elif input_image_format =='image':
         base64_encoded_image_string = request_params['image']
         BGR_image =Image.open(BytesIO(base64.b64decode(base64_encoded_image_string)))
         RGB_image=cv2.cvtColor(np.array(BGR_image), cv2.COLOR_BGR2RGB)
         detected_objets=camera.detect(RGB_image, input_image_format)
    else:
        return "Invalid request"

    return format_result(output_format,output_minify,RGB_image,detected_objets)


def format_result(output_format,output_minify,RGB_image,detected_objets):
    """
     format the resutls according to user request.It return a json file
     containing the list of objects detected in the image or retuen the
     original image with bounding box around the detected objects.
    :param output_format:
    :param RGB_image:
    :param detected_objets:
    :return:
    """
    if output_format=='json':
        return jasonify(detected_objets,output_minify)
    elif output_format=='image':
        base64Image=str(generate_output_image(RGB_image, detected_objets))
        base64Image=base64Image[2:]
        base64Image=base64Image[:-1]
        response={
            "image":base64Image,
            "objects":jasonify(detected_objets,True)
        }
        return json.dumps(response);
        #return generate_output_image(RGB_image, detected_objets)


def generate_output_image(original_image, detected_objects):
    overlayed_image =draw_objects_boundaries(original_image, detected_objects)
    RGB_image = cv2.cvtColor(np.array(overlayed_image), cv2.COLOR_BGR2RGB)
    return encode_base64(RGB_image)


def draw_objects_boundaries (input_image, detected_objects):
    """
    Draws the bouding box around the detected objects
    and overlay that on the original image
    :param input_image:
    :param detected_objects:
    :return: original image with boundary box drawn around
     the detected objects.
    """
    image={}
    for i in range(len(detected_objects)):
        topLeft = (detected_objects[i]['topleft']['x'], detected_objects[i]['topleft']['y'])
        bottomRight = (detected_objects[i]['bottomright']['x'], detected_objects[i]['bottomright']['y'])
        label = detected_objects[i]['label']
        image = cv2.rectangle(input_image, topLeft, bottomRight, (0, 255, 0), 2)
        image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        image = cv2.putText(input_image, label, topLeft, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    return image

def encode_base64(image):
    retval, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer)


def jasonify(input,output_minify):
    results=pd.Series(input).to_json(orient='values');
    if not output_minify:
        return results
    minified={}
    results=json.loads(results)
    for result in results:
        label=result['label']
        if label in minified:
            minified[label]= minified[label]+1
        else:
            minified[label]=1
    return json.dumps(minified)

def get_image_format(params):
    if 'http' in params['image']:
        return 'url'
    else:
        return 'image'

def validate_request(request_params):
    output_format = request_params['output']
    if output_format not in ['json','image']:
        return False



if __name__=="__main__":
    app.run()



