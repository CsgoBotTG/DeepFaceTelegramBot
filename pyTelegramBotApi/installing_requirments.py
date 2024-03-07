import os

print('requirments')
os.system('pip install -r ' + '\\'.join([item[::-1] for item in os.path.abspath(__file__)[::-1].split('\\')[1:]][::-1]) + '/requirments.txt')

import ultralytics
from TelegramBotNames import models_path

print('yolo')
object_detection_yolo_models = [
    'yolov8n.pt',
    'yolov8s.pt',
    'yolov8m.pt',
    'yolov8l.pt',
    'yolov8x.pt',
]
[ultralytics.YOLO(models_path + model + '.pt') for model in object_detection_yolo_models]
[ultralytics.YOLO(models_path + model + '-seg.pt') for model in object_detection_yolo_models]
[ultralytics.YOLO(models_path + model + '-pose.pt') for model in object_detection_yolo_models]

from deepface import DeepFace

print('deepface')
models_deepface = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
]
detector_backends = [
    'opencv', 
    'retinaface',
    'mtcnn',
    'ssd',
    'dlib',
    'mediapipe',
    'yolov8',
]
[DeepFace.extract_faces('../images/Emotion.png', detector_backend=detector_backend) for detector_backend in detector_backends]
[DeepFace.verify('../images/Harry1.jpg', '../images/Harry2.jpg', model_name=model) for model in models_deepface]