# encoding:utf-8
from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()
model_path = "/Users/rensike/Resources/models/imageai/resnet50_coco_best_v2.0.1.h5"

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(model_path)
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "images/test.jpeg"),  output_image_path=os.path.join(execution_path , "output/test_detection.jpg"), extract_detected_objects=True)

print(len(detections))

for eachObject in detections[0]:
    print(eachObject["name"], " : ", eachObject["percentage_probability"])

