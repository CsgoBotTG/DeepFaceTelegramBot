from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton, ReplyKeyboardMarkup, ReplyKeyboardRemove
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup

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
pose_yolo_models = [
    'yolov8n-pose.pt',
    'yolov8s-pose.pt',
    'yolov8m-pose.pt',
    'yolov8l-pose.pt',
    'yolov8x-pose.pt',
]


text_start_menu = [
    ["Find faces in a photo🔎", "Verify faces🤓🥸", "Analyze face☹️😀"],
    ["Object detection🕵️", "Segmentation✒️", "Pose people🧑‍🤝‍🧑"],
    ["Settings⚙️", 'Finish Bot👊'],
]
buttons_start_menu = [[KeyboardButton(text=text) for text in list_text] for list_text in text_start_menu]
start_menu = ReplyKeyboardMarkup(keyboard=buttons_start_menu)

text_settings_menu = [
    ["Detector backend"],
    ["Model Neural Network for analyze face"],
    ["Object Detection Yolo Model"],
    ["Segmentation Yolo Model"],
    ["Pose Yolo Model"],
    ["Info"],
]
buttons_settings_menu = [[InlineKeyboardButton(text=text, callback_data=text) for text in list_text] for list_text in text_settings_menu]
settings_menu = InlineKeyboardMarkup(inline_keyboard=buttons_settings_menu)

buttons_detector_backend = [[InlineKeyboardButton(text=text, callback_data=text)] for text in detector_backends_models]
detector_backend_menu = InlineKeyboardMarkup(inline_keyboard=buttons_detector_backend)

buttons_deepface_analyze = [[InlineKeyboardButton(text=text, callback_data=text)] for text in deepface_analyze_models]
deepface_analyze_menu = InlineKeyboardMarkup(inline_keyboard=buttons_deepface_analyze)

buttons_object_detection_yolo = [[InlineKeyboardButton(text=text, callback_data=text)] for text in object_detection_yolo_models]
object_detection_yolo_menu = InlineKeyboardMarkup(inline_keyboard=buttons_object_detection_yolo)

buttons_segmentation_yolo = [[InlineKeyboardButton(text=text, callback_data=text)] for text in segmentation_yolo_models]
segmentation_yolo_menu = InlineKeyboardMarkup(inline_keyboard=buttons_segmentation_yolo)

buttons_pose_yolo = [[InlineKeyboardButton(text=text, callback_data=text)] for text in pose_yolo_models]
pose_yolo_menu = InlineKeyboardMarkup(inline_keyboard=buttons_pose_yolo)

class FindFacesForm(StatesGroup):
    image = State()

class VerifyFaceForm(StatesGroup):
    image_base = State()
    image_verify = State()

class AnalyzeFaceForm(StatesGroup):
    image = State()

class ObjectDetectionYoloForm(StatesGroup):
    image = State()

class SegmentationYoloForm(StatesGroup):
    image = State()

class PoseYoloForm(StatesGroup):
    image = State()