from telebot.types import ReplyKeyboardMarkup, KeyboardButton
from telebot.util import quick_markup

models_path = r'../../weightsYolo/'

deepface_analyze_models = [
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
detector_backends_models = [
    'opencv', 
    'retinaface',
    'mtcnn',
    'ssd',
    'dlib',
    'mediapipe',
    'yolov8',
]
object_detection_yolo_models = [
    'yolov8n.pt',
    'yolov8s.pt',
    'yolov8m.pt',
    'yolov8l.pt',
    'yolov8x.pt',
]
segmentation_yolo_models = [
    'yolov8n-seg.pt',
    'yolov8s-seg.pt',
    'yolov8m-seg.pt',
    'yolov8l-seg.pt',
    'yolov8x-seg.pt',
]


text_start_menu = [
    ["Find faces in a photo🔎", "Verify faces🤓🥸", "Analyze face☹️😀"],
    ["Object detection🕵️", "Segmentation✒️"],
    ["Settings⚙️"],
]
start_menu = ReplyKeyboardMarkup()
[start_menu.add(*[KeyboardButton(text) for text in texts]) for texts in text_start_menu]

text_settings_menu = [
    "Detector backend",
    "Model Neural Network for analyze face",
    "Object Detection Yolo Model",
    "Segmentation Yolo Model",
    "Info",
]
settings_menu = quick_markup({i:{'callback_data': i} for i in text_settings_menu})

detector_backend_menu = quick_markup({i:{'callback_data': i} for i in detector_backends_models})

deepface_analyze_menu = quick_markup({i:{'callback_data': i} for i in deepface_analyze_models})

object_detection_yolo_menu = quick_markup({i:{'callback_data': i} for i in object_detection_yolo_models})

segmentation_yolo_menu = quick_markup({i:{'callback_data': i} for i in segmentation_yolo_models})