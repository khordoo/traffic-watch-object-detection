import simplejson
from flask import Flask, request, abort
from flask_cors import cross_origin
import werkzeug
from services import ImageAnalysisService, Database, HistoricalImagePoinst
from config import Config

app = Flask(__name__)
config = Config()
image_service = ImageAnalysisService()
database = Database(config)
historical_detections = HistoricalImagePoinst(database)


@app.route('/analysis', methods=['POST', 'GET'])
@cross_origin()
def index():
    if request.method == 'GET':
        return """<h1>Yep! It works!</h1> \n Send POST request for object detection."""

    if request.method == 'POST':
        try:
            if not request.json:
                abort(400)
            prediction = image_service.detect(request)
            prediction['detections'] = simplejson.dumps(prediction['detections'])
            response = app.response_class(
                response=simplejson.dumps(prediction),
                status=200,
                mimetype='application/json',
            )
            return response
        except werkzeug.exceptions.BadRequest as e:
            abort(400, str(e))


@app.route('/historical', methods=['POST', 'GET'])
@cross_origin()
def objects_history():
    if request.method == 'GET':
        return """<h1>Yep! It works!</h1> \n Send POST request for getting the historical object locations."""
    if request.method == 'POST':
        try:
            if not request.json:
                abort(400)
            response = historical_detections.draw(request)
            response = app.response_class(
                response=simplejson.dumps(response),
                status=200,
                mimetype='application/json',
            )
            return response
        except werkzeug.exceptions.BadRequest as e:
            abort(400, str(e))


if __name__ == "__main__":
    app.run()
