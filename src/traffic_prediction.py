import psycopg2
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.models import load_model
from datetime import timedelta
from dateutil import tz
from dateutil.parser import parse


class TrafficDatabaseConnector:
    def __init__(self, host, database, username, password):
        try:
            conn = psycopg2.connect(host=host, database=database, user=username, password=password)
            self.cursor = conn.cursor()
        except Exception as err:
            print(f'Failed to connect to Postgres database: {err}')

    def fetch_recent_history(self, camera_id, number_of_records, aggregation_minutes_interval=15):
        """Fetches the  top {} most recent records for the {entity} from the database"""
        query = f"SELECT   date_trunc('hour', time) + (((date_part('minute', time)::integer / {aggregation_minutes_interval}::integer) * {aggregation_minutes_interval}::integer) || ' minutes')::interval AS time_interval, " \
                f"avg(count) FROM public.count where camera_id={camera_id} and label in ('car','bus','truck')  GROUP BY time_interval order by time_interval desc   limit {number_of_records};"
        self.cursor.execute(query)
        return self.cursor.fetchall()


class TrafficPrediction:
    def __init__(self, database, model_path, scaler_path, history_time_steps=24):
        self.database = database
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.TIME_SHIFT_STEPS = history_time_steps
        self.TIME_INTERVALS = timedelta(minutes=15)
        self.TIME_ZONES = 'America/Edmonton'

    def predict(self, camera_id, prediction_steps):
        inputs, historical_dates = self._prepare_inputs(camera_id, self.TIME_SHIFT_STEPS)
        X_train_hourOfDay_pred, X_train_dayOfWeek_pred, X_train_dayOfYear_pred, X_train_pred = inputs
        prediction_dates = self.create_dates(historical_dates, prediction_steps)
        predictions = []
        for i in range(prediction_steps):
            prediction = self.model.predict([X_train_hourOfDay_pred, X_train_dayOfWeek_pred, X_train_dayOfYear_pred,
                                             X_train_pred[:, :, :self.TIME_SHIFT_STEPS]])
            predictions.append(self.scaler.inverse_transform(prediction)[0][0])
            X_train_pred = np.insert(X_train_pred, 0, prediction[0][0], axis=2)

        pred_response = []
        for date, prediction in zip(prediction_dates, predictions):
            pred_response.append({"time": date.isoformat(), "count": float(prediction)})
        historical_hourly = self.get_hourly_historical(camera_id, 24)
        return {"prediction": pred_response, "historical": historical_hourly}

    def _prepare_inputs(self, camera_id, steps):
        # records = self.database.fetch_recent_history(camera_id, 2 * self.TIME_SHIFT_STEPS)
        records = self.database.fetch_recent_history(camera_id, 2 * self.TIME_SHIFT_STEPS)
        historical = pd.DataFrame(records, columns=['phenomenonTime', 'result'])
        historical['result'] = historical['result'].astype(float)
        historical.set_index(pd.DatetimeIndex(historical['phenomenonTime']), inplace=True)
        historical.drop(['phenomenonTime'], inplace=True, axis=1)

        historical[['result']] = self.scaler.transform(historical['result'].values.reshape(-1, 1))
        print(historical.head())
        train_pred, y_train_pred = self.get_timeseries(historical.copy(), self.TIME_SHIFT_STEPS, 'result')
        # We only want a single row for the last point to input into model
        # A long history were only fetched to be able to create the past events for out last (most recent point)
        # So after beuiling the historical aray for our last point [ 0,12,,] , the remaining older points are not useful
        # for our predtion task, thought if it was a training tsk they could be used as well

        # Training
        data = historical.copy()
        data.drop(['result'], inplace=True, axis=1)
        data['hourOfDay'] = data.index.hour
        data['dayOfWeek'] = data.index.dayofweek
        data['dayOfYear'] = data.index.dayofyear
        train_cat_pred = self.get_categorical(data, self.TIME_SHIFT_STEPS)
        x_train_pred = np.array(train_pred['result_merged'].values.tolist()[0]).reshape(-1, 1, self.TIME_SHIFT_STEPS)
        x_train_hourOfDay_pred = np.array(train_cat_pred['hourOfDay_merged'].values.tolist()[0]).reshape(-1,
                                                                                                         self.TIME_SHIFT_STEPS)
        x_train_dayOfWeek_pred = np.array(train_cat_pred['dayOfWeek_merged'].values.tolist()[0]).reshape(-1,
                                                                                                         self.TIME_SHIFT_STEPS)
        x_train_dayOfYear_pred = np.array(train_cat_pred['dayOfYear_merged'].values.tolist()[0]).reshape(-1,
                                                                                                         self.TIME_SHIFT_STEPS)
        return [x_train_hourOfDay_pred, x_train_dayOfWeek_pred, x_train_dayOfYear_pred, x_train_pred], train_pred.index

    def get_timeseries(self, data, steps, target_column):
        y = None
        df = None
        for column in data.columns.tolist():
            df = data[[column]].copy()
            for i in range(1, steps + 1):
                df[f'{column}{i}'] = df[column].shift(-i)
            df.dropna(inplace=True, axis=0)

            if column == target_column:
                y = df[f'{column}{steps}'].values
            df.drop([f'{column}{steps}'], inplace=True, axis=1)
            df[f'{column}_merged'] = df.values.tolist()
            df.drop([f'{column}'], inplace=True, axis=1)
            for i in range(1, steps):
                df.drop([f'{column}{i}'], inplace=True, axis=1)

        return df, y

    def get_categorical(self, data, steps):
        merged = None
        for column in data.columns.tolist():
            df = data[[column]]
            for i in range(1, steps + 1):
                df[f'{column}{i}'] = df[column].shift(-i)
            df.dropna(inplace=True, axis=0)
            df.drop([f'{column}{steps}'], inplace=True, axis=1)
            df[f'{column}_merged'] = df.values.tolist()
            df.drop([f'{column}'], inplace=True, axis=1)
            for i in range(1, steps):
                df.drop([f'{column}{i}'], inplace=True, axis=1)

            if merged is None:
                merged = df.copy()
            else:
                merged[f'{column}_merged'] = df[[f'{column}_merged']]

        return merged

    def create_dates(self, hist_dates, prediction_steps):
        most_recent_date = hist_dates[0]
        most_recent_date = most_recent_date.replace(tzinfo=tz.gettz(self.TIME_ZONES))
        pred_dates = []
        for i in range(1, prediction_steps + 1):
            pred_dates.append(most_recent_date + i * self.TIME_INTERVALS)
        return pred_dates

    def get_hourly_historical(self, camera_id, limit=24):
        records = self.database.fetch_recent_history(camera_id=camera_id, number_of_records=limit,
                                                     aggregation_minutes_interval=30)
        records = [
            {'time': record[0].replace(tzinfo=tz.gettz(self.TIME_ZONES)).isoformat(), 'count': float(record[1])}
            for record in records]
        return records
