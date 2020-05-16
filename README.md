### Smart Traffic and Pedestrian movement analyzer 

This repository contains the code that combines the CNN and RNN to capture the traffic pattern acroos the City of Calgary and 
make a prediction about the volume of the traffic across the city.

Here is an screen shot of the application UI:
![image](https://user-images.githubusercontent.com/32692718/82127114-7a2a2300-976e-11ea-9fcd-feb20a4bbb78.png)

## Motivation:
Traffic congestion primarily occurs due to unknown factors such as bad weather conditions, unexpected vehicular failure or a road accident. So, a continuous evaluation of the road traffic needs to be done to determine the congestion free paths. Unlike traditional approaches for determining traffic flow (e.g., hose counts and manual counts) and conventional sensors used in road traffic monitoring (e.g., auto scope and loop detectors), cameras provide the best technology to acquire the data in real- time due to their higher sensing range. Cities usually shares their traffic data with the public so that citizens can be informed of what is happening in their city. Such information also helps officials make smart traffic management decisions, such as when to implement (1) traffic-calming measures, (2) walking and cycling improvements, (3) traffic and parking regulations, and (4) bus stop locations.


## Architecture 
Traffic Watch is built using free and open source software, open standards, and open data - YOLO, TensorFlow 2.0, NodeJS and Express, VueJS, Vuetify, and Mapbox GL JS are used to create the system components. It collects camera images from the City of Calgary’s open data website every 5 minutes, analyzes them using a machine learning models built by YOLO and TensorFlow 2.0, and displays results on a map. Figure below shows Traffic Watch's overall architecture.
![image](https://res.cloudinary.com/devpost/image/fetch/s--9Dc_3Mz0--/c_limit,f_auto,fl_lossy,q_auto:eco,w_900/https://www.dropbox.com/s/7t9zry5ybdqsiy5/traffic_watch_revised_architecture.png%3Fdl%3D1)

## Anomaly detection
The Application can also use the historical detection data to create a cloud of the movement patterns and identify the
anomalies in the movement data. It can help detect dangerous driving patterns or unusual crossing locations in streets.

Here is an example of vehicle movement pattern for a specific location:

![image](https://user-images.githubusercontent.com/32692718/82127375-4e0fa180-9770-11ea-9457-1f1eb5cc7b4c.png)
