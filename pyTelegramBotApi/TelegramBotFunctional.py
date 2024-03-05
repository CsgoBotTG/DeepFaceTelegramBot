import os
import cv2
import numpy as np

import telebot
import DeepFaceFunctionality
import ObjectDetectionFunctionality


def TelegramFindFacesFunctional(bot: telebot.TeleBot, message: telebot.types.Message, detector_backend='opencv'):
    file_info = bot.get_file(message.photo[len(message.photo)-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    image_bytes_arr = np.fromstring(downloaded_file, np.uint8)
    image = cv2.imdecode(image_bytes_arr, cv2.IMREAD_COLOR)

    bot.send_message(message.chat.id, 'Got image')
    bot.send_message(message.chat.id, 'Starting recognition')

    faces = DeepFaceFunctionality.FacesInPhoto(image, detector_backend=detector_backend)

    for index, face in enumerate(faces):
        area = face['facial_area']

        cv2.rectangle(image, (area['x'], area['y']), (area['x'] + area['w'], area['y'] + area['h']), (255, 69, 69), 4)

        face_image = image[area['y']:area['y']+area['h'], area['x']:area['x'] + area['w']]
        face_image = cv2.imencode('.jpg', face_image)[1].tostring()

        bot.send_message(message.chat.id, f'Face {index + 1}:')
        bot.send_photo(message.chat.id, face_image)
                
    bot.send_message(message.chat.id, 'Highlighting faces...')

    image = cv2.imencode('.jpg', image)[1].tostring()
    bot.send_photo(message.chat.id, image)

    bot.send_message(message.chat.id, f'[+] Detector Backend: {detector_backend}')


def TelegramVerifyFace1(bot: telebot.TeleBot, message: telebot.types.Message):
    file_info = bot.get_file(message.photo[len(message.photo)-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    image_bytes_arr = np.fromstring(downloaded_file, np.uint8)
    image = cv2.imdecode(image_bytes_arr, cv2.IMREAD_COLOR)

    bot.send_message(message.chat.id, 'Got image')
    return image


def TelegramVerifyFace2(bot: telebot.TeleBot, message: telebot.types.Message, image_base: np.ndarray, detector_backend='opencv', model_name='VGG-Face'):
    file_info = bot.get_file(message.photo[len(message.photo)-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    image_bytes_arr = np.fromstring(downloaded_file, np.uint8)
    image_verify = cv2.imdecode(image_bytes_arr, cv2.IMREAD_COLOR)

    bot.send_message(message.chat.id, 'Got image')
    bot.send_message(message.chat.id, 'Starting recognition...')

    result = DeepFaceFunctionality.VerifyInPhotos(image_base, image_verify, detector_backend=detector_backend, model_name=model_name)
    result_msg = ''

    result_msg += f'[+] Verified: {result["verified"]}\n'
    result_msg += f'[+] Distance: {round(result["distance"], 2)}\n'
    result_msg += f'[+] Threshhold: {round(result["threshold"], 2)}\n'
    result_msg += f'[+] Model: {model_name}\n'
    result_msg += f'[+] Detector Backend: {detector_backend}'
                
    bot.send_message(message.chat.id, 'Finished')
    bot.send_message(message.chat.id, result_msg)

                
    bot.send_message(message.chat.id, 'Highlighting faces....')

    cv2.rectangle(image_base, (result['facial_areas']['img1']['x'], result['facial_areas']['img1']['y']), (result['facial_areas']['img1']['x'] + result['facial_areas']['img1']['w'], result['facial_areas']['img1']['y'] + result['facial_areas']['img1']['h']), (255, 69, 69), 3)
    cv2.rectangle(image_verify, (result['facial_areas']['img2']['x'], result['facial_areas']['img2']['y']), (result['facial_areas']['img2']['x'] + result['facial_areas']['img2']['w'], result['facial_areas']['img2']['y'] + result['facial_areas']['img2']['h']), (255, 69, 69), 3)

    if image_base.shape[0] * image_base.shape[1] > image_verify.shape[0] * image_verify.shape[1]:
        image_base = cv2.resize(image_base, image_verify.shape[:2][::-1])
    else:
        image_verify = cv2.resize(image_verify, image_base.shape[:2][::-1])

    vis = np.concatenate((image_base, image_verify), axis=1)
    vis = cv2.imencode('.jpg', vis)[1].tostring()

    bot.send_photo(message.chat.id, vis)


def TelegramAnalyzeFace(bot: telebot.TeleBot, message: telebot.types.Message, markup: telebot.types.InlineKeyboardMarkup, detector_backend='opencv'):
    file_info = bot.get_file(message.photo[len(message.photo)-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    image_bytes_arr = np.fromstring(downloaded_file, np.uint8)
    image = cv2.imdecode(image_bytes_arr, cv2.IMREAD_COLOR)

    bot.send_message(message.chat.id, 'Got image')
    bot.send_message(message.chat.id, 'Starting recognition...')

    analyze = DeepFaceFunctionality.AnalyzeFaceEmotionInPhoto(image, detector_backend=detector_backend)[0]

    gender = 'Woman'
    if analyze.get('gender').get('Man') > analyze.get('gender').get('Woman'):
        gender = 'Man'
                
    result_msg = ''
    result_msg += f'[+] Age: {analyze.get("age")}\n'
    result_msg += f'[+] Gender: {gender}\n'
    result_msg += f'[+] Race:\n'

    for k, v in analyze.get('race').items():
        result_msg += f'\t\t\t{k} - {round(v, 2)}%\n'
                
    result_msg += f'[+] Dominant race: {analyze["dominant_race"]}\n'
    result_msg += f'[+] Emotions:\n'

    for k, v in analyze.get('emotion').items():
        result_msg += f'\t\t\t{k} - {round(v, 2)}%\n'
                
    result_msg += f'[+] Dominant emotion: {analyze["dominant_emotion"]}\n'
    result_msg += f'[+] Detector Backend: {detector_backend}'
                
    bot.send_message(message.chat.id, 'Finished')
    bot.send_message(message.chat.id, result_msg)
                
    bot.send_message(message.chat.id, 'Highlighting faces...')

    face_region = analyze['region']
    face = image[face_region['y']:face_region['y']+face_region['h'], face_region['x']:face_region['x']+face_region['w']]
    face_str = cv2.imencode('.jpg', face)[1].tostring()

    bot.send_photo(message.chat.id, face_str, reply_markup=markup)


def TelegramObjectDetection(bot: telebot.TeleBot, message: telebot.types.Message, object_detection_yolo: str):
    bot.send_message(message.chat.id, "Got Image")
    bot.send_message(message.chat.id, "Starting recognition...")

    file_info = bot.get_file(message.photo[len(message.photo)-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    image_bytes_arr = np.fromstring(downloaded_file, np.uint8)
    image = cv2.imdecode(image_bytes_arr, cv2.IMREAD_COLOR)

    if not os.path.exists("weightsYolo/" + object_detection_yolo):
        bot.send_message(message.chat.id, "Didn't find model. Wait few minutes. Downloading...")

    result, image = ObjectDetectionFunctionality.object_detect_with_highligth(image, object_detection_yolo)                

    bot.send_message(message.chat.id, "Finished")

    image_str = cv2.imencode('.jpg', image)[1].tostring()
    bot.send_photo(message.chat.id, image_str)
