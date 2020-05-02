from darkflow.net.build import TFNet


class YOLO:
    """
    Python Wrapper for YOLO
    """
    displayImageWindow = False
    threshold = 0.12
    options = {
        'model': 'cfg/yolo.cfg',
        'load': 'weights/yolov2.weights',
        'threshold': threshold
    }

    def __init__(self):
        self.tfnet = TFNet(self.options)

    def detect(self, image):
        return self.tfnet.return_predict(image)
