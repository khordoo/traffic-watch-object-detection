import os
import simplejson
from flask import Flask, request, abort
from flask_cors import CORS
import werkzeug
from src.services import ImageAnalysisService
from src.services import HistoricalImagePoinst
from src.services import Database
from src.traffic_prediction import TrafficDatabaseConnector
from src.traffic_prediction import TrafficPrediction
from src.config import Config
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)
CORS(app)
conf = Config()

session = tf.Session()
keras.backend.set_session(session)

base_dir = os.path.join(os.path.dirname(__file__), 'save')
database_pg = TrafficDatabaseConnector(host=conf.postgres_host, database=conf.postgres_database_name,
                                       username=conf.postgres_username, password=conf.postgres_password)
predictor = TrafficPrediction(database=database_pg, model_path=base_dir + '/model.h5',
                              scaler_path=base_dir + '/scaler.joblib',
                              history_time_steps=conf.historical_time_window_size)

image_service = ImageAnalysisService()
database = Database(conf)
historical_detections = HistoricalImagePoinst(database)


@app.route('/analysis', methods=['POST', 'GET'])
def index():
    """Performs a real time object detection"""
    if request.method == 'GET':
        return """<h1>Yep! It works!</h1> \n Send POST request for object detection."""

    if request.method == 'POST':
        try:
            if not request.json:
                abort(400)
            payload = request.get_json(force=True)
            prediction = image_service.detect(payload)
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
def objects_history():
    """Create historical point cloud using coordinates of detected objects point in a location"""
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


@app.route('/predict', methods=['POST', 'GET'])
def traffic_prediction():
    """Performs the time series analysis to predict the traffic flow in the future"""
    if request.method == 'GET':
        return 'Welcome to Calgary Traffic Flow Prediction Service'

    if request.method == 'POST':
        if not request.json:
            abort(400)
        payload = request.json

        with session.as_default():
            with session.graph.as_default():
                prediction = predictor.predict(camera_id=payload['cameraId'], prediction_steps=payload['steps'])

        response = app.response_class(
            response=simplejson.dumps(prediction),
            status=200,
            mimetype='application/json',
        )
        return response
