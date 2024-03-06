import os
import cvzone
import numpy as np

from TelegramBotFunctionsHelper import *
from DeepFaceFunctionality import *
from ObjectDetectionFunctionality import *

from telebot import TeleBot
from telebot.types import Message


def telegram_find_face_functional(
        bot: TeleBot, 
        message: Message, 
        detector_backend: str = 'yolov8'
    ) -> None:
    """
    Find faces from message.

    :param bot: telebot.Telebot. Get bot to download and send images
    :param message: telebor.types.Message. Get message to get chat_id
    :param detector_backend: str. Used detector backed
    """

    image = get_image_from_message(bot, message)

    bot.send_message(message.from_user.id, 'Got image')
    bot.send_message(message.from_user.id, 'Starting recognition')

    faces = faces_in_photo(image, detector_backend)
    for index, face in enumerate(faces):
        area = face['facial_area']
        x, y, w, h = area['x'], area['y'], area['w'], area['h']

        cvzone.cornerRect(image, (x, y, w, h), l=9)
        cvzone.putTextRect(image, f'Face {index + 1}', (max(0, x), max(35, y)), scale=0.75, thickness=1, offset=3)
        face_image = image[y:y+h, x:x+w]

        send_image(bot, message, face_image, f'Face {index + 1}')

    bot.send_message(message.from_user.id, 'Highlighting faces...')
    send_image(bot, message, image)
    bot.send_message(message.from_user.id, f'[+] Detector Backend: {detector_backend}')

    return None


def telegram_verify_faces_functional(
        bot: TeleBot, 
        message: Message, 
        image_base: np.ndarray, 
        detector_backend='opencv', 
        model_name='VGG-Face'
    ) -> None:
    """
    Verifing 2 faces from message

    :param bot: telebot.TeleBot. Get bot to download and send
    :param message: telebot.types.Message. Get message for chat_id
    :param image_base: np.ndarray. image to verify second photo
    :param detector_backend: str. Used detector backend
    :param model_name: str. Used model of NN analizing face
    """

    image_verify = get_image_from_message(bot, message)

    bot.send_message(message.from_user.id, 'Got image')
    bot.send_message(message.from_user.id, 'Starting recognition...')

    result = verify_in_photo(image_base, image_verify, detector_backend, model_name)

    result_msg =  f'[+] <b>Verified</b>: {result["verified"]}\n'
    result_msg += f'[+] <b>Distance</b>: {round(result["distance"], 2)}\n'
    result_msg += f'[+] <b>Threshhold</b>: {round(result["threshold"], 2)}\n'
    result_msg += f'[+] <b>Model</b>: {model_name}\n'
    result_msg += f'[+] <b>Detector Backend</b>: {detector_backend}'
            
    bot.send_message(message.from_user.id, 'Finished')
    bot.send_message(message.from_user.id, result_msg, parse_mode='HTML')
    bot.send_message(message.from_user.id, 'Highlighting faces....')

    x1, y1, w1, h1 = result['facial_areas']['img1']['x'], result['facial_areas']['img1']['y'], result['facial_areas']['img1']['w'], result['facial_areas']['img1']['h'] 
    x2, y2, w2, h2 = result['facial_areas']['img2']['x'], result['facial_areas']['img2']['y'], result['facial_areas']['img2']['w'], result['facial_areas']['img2']['h'] 

    cvzone.cornerRect(image_base, (x1, y1, w1, h1), l=9)
    cvzone.cornerRect(image_verify, (x2, y2, w2, h2), l=9)

    send_image(bot, message, image_base, 'Base Image')
    send_image(bot, message, image_verify, 'Verify Image')

    return None


def telegram_analyze_face_functional(
        bot: TeleBot, 
        message: Message, 
        detector_backend='opencv'
    ) -> None:
    """
    Analyze face: emotions, age, race, gender from message

    :param bot: telebot.TeleBot. Get bot to download and send
    :param message: telebot.types.Message. Get message for chat_id
    :param detector_backend: str. Used detector_backends
    """

    image = get_image_from_message(bot, message)

    bot.send_message(message.from_user.id, 'Got image')
    bot.send_message(message.from_user.id, 'Starting recognition...')

    analyze_faces = analyze_face_in_photo(image, detector_backend)
    for index, analyze in enumerate(analyze_faces):
        gender = 'Woman' if analyze.get('gender').get('Man') < analyze.get('gender').get('Woman') else 'Man'

        result_msg =  f'Face {index + 1}\n'
        result_msg += f'[+] <b>Age</b>: {analyze["age"]}\n'
        result_msg += f'[+] <b>Gender</b>: {gender}\n'
        result_msg +=  '[+] <b>Race</b>:\n'

        for k, v in analyze.get('race').items():
            result_msg += f'\t\t\t<b>{k}</b> - <u>{round(v, 2)}%</u>\n'

        result_msg += f'[+] <b>Dominant race</b>: {analyze["dominant_race"]}\n'
        result_msg +=  '[+] <b>Emotions</b>:\n'

        for k, v in analyze.get('emotion').items():
            result_msg += f'\t\t\t<b>{k}</b> - <u>{round(v, 2)}%</u>\n'

        result_msg += f'[+] <b>Dominant emotion</b>: {analyze["dominant_emotion"]}\n'
        result_msg += f'[+] <b>Detector Backend</b>: {detector_backend}'

        bot.send_message(message.chat.id, 'Finished')
        bot.send_message(message.chat.id, result_msg, parse_mode='HTML')

        face_region = analyze['region']
        x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']
        face = image[y:y+h, x:x+w]
        send_image(bot, message, face, f'Face {index + 1}')
        cvzone.putTextRect(image, f'Face {index + 1}', (max(0, x), max(35, y)), scale=0.75, thickness=1, offset=3)
        cvzone.cornerRect(image, (x, y, w, h), l=9)
    
    bot.send_message(message.chat.id, 'Highlighting faces...')
    send_image(bot, message, image, 'Highlitghed')

    return None


def telegram_object_detection_functional(
        bot: TeleBot, 
        message: Message, 
        object_detection_yolo: str
    ) -> None:
    """
    Object detection from message

    :param bot: telebot.TeleBot. Get bot to download and send
    :param message: telebot.types.Message. Get message for chat_id
    :param object_detection_yolo: str. Yolov8 object detection model
    """

    image = get_image_from_message(bot, message)

    bot.send_message(message.from_user.id, "Got Image")
    bot.send_message(message.from_user.id, "Starting recognition...")

    if not os.path.exists(models_path + object_detection_yolo):
        bot.send_message(message.from_user.id, "Didn't find model. Wait few seconds. Downloading...")

    _, image = object_detect_with_highligth(image, object_detection_yolo)                

    bot.send_message(message.from_user.id, "Finished")
    bot.send_message(message.from_user.id, "Highligthing image")
    send_image(bot, message, image, 'Highlitghed Image')

    return None


def telegram_segmentation_functional(
        bot: TeleBot,
        message: Message,
        segmentation_yolo: str = 'yolov8x-seg.pt'   
    ) -> None:
    """
    Object detection from message

    :param bot: telebot.Bot. Get bot to download and send
    :param message: telebot.types.Message. Get message for chat_id
    :param segmentation_yolo: str. Yolov8 object detection model
    """

    image = get_image_from_message(bot, message)
    
    bot.send_message(message.from_user.id, "Got Image")
    bot.send_message(message.from_user.id, "Starting recognition...")

    if not os.path.exists(models_path + segmentation_yolo):
        bot.send_message(message.from_user.id, "Didn't find model. Wait few seconds. Downloading...")
    
    _, image = segmentation_detect_with_higligth(image, segmentation_yolo)

    bot.send_message(message.from_user.id, "Finished")
    bot.send_message(message.from_user.id, "Highligthing image")
    send_image(bot, message, image, 'Highlitghed Image')

    return None