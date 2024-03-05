import os
import ultralytics
from deepface import DeepFace

print('requirments')
os.system('pip install -r requirments.txt')

print('yolo')
object_detection_yolo_models = [
    'yolov8n.pt',
    'yolov8s.pt',
    'yolov8m.pt',
    'yolov8l.pt',
    'yolov8x.pt'
]
[ultralytics.YOLO("weightsYolo/" + model) for model in object_detection_yolo_models]

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
    'yolov8'
]
[DeepFace.extract_faces('images/Emotion.png', detector_backend=detector_backend) for detector_backend in detector_backends]
[DeepFace.verify('images/Harry1.jpg', 'images/Harry2.jpg', model_name=model) for model in models_deepface]