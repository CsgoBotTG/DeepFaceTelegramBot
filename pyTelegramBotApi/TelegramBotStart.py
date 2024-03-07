import json

from TelegramBotNames import *
from TelegramBotFunctions import *
from TelegramBotFunctionsHelper import *

from telebot import TeleBot
from telebot.types import Message, CallbackQuery


def start_bot(
        token: str, 
        to_log: bool = True
    ) -> None:
    """
    Starting bot with your token

    :param token: Your token. Could get in @BotFather.
    :param to_log: Bool. Printing bot's token and address.
    """

    # init
    bot = TeleBot(token)
    storage = {
        'deepface_analyze': deepface_analyze_models[-1],
        'detector_backend': detector_backends_models[-1],
        'object_detection_yolo': object_detection_yolo_models[-1],
        'segmentation_yolo': segmentation_yolo_models[-1],
        'pose_yolo': pose_yolo_models[-1],
        'first_message': True,
        'session': False,
        'image_base': None, # verify
    }

    # start/help menu
    @bot.message_handler(commands=['start', 'help'])
    @session_waiter(bot, storage)
    def start(message: Message):
        if storage['first_message']:
            bot.send_animation(message.from_user.id, 'https://media1.tenor.com/m/5hKPyupKGWMAAAAC/robot-hello.gif')
            bot.send_message(message.from_user.id, f"Hello, {message.from_user.first_name}! I'm BOT that working on YOLOv5, Deepface, Tensorflow and pyTelegramBotAPI!")

            storage['first_message'] = False

        bot.send_message(message.from_user.id, 
                                "My functionality are: " + \
                                "\n1. DeepFace\n\t\t\t\t1. Find faces in a photo;\n\t\t\t\t" + \
                                "2. Verify faces several photos;" + \
                                "\n\t\t\t\t3. Analyze facial emotions;\n" + \
                                "2. YoloV8\n\t\t\t\t1. Object Detection;\n\t\t\t\t" + \
                                "2. Segmentation;\n\t\t\t\t" + \
                                "3. Pose people;", 
                         reply_markup=start_menu)
    

    # find faces
    @bot.message_handler(func=lambda message: message.text == "Find faces in a photo🔎")
    @session_waiter(bot, storage)
    def get_find_faces(message: Message):
        sent_message = bot.send_message(message.from_user.id, 'Send photo where you wanna highligth faces')
        bot.register_next_step_handler(sent_message, find_faces) # to get photo


    @session_waiter(bot, storage)
    def find_faces(message: Message):
        if message.content_type != 'photo': # exceptions
            bot.send_message(message.from_user.id, "It isn't photo...")
        else:
            telegram_find_face_functional(
                bot=bot, 
                message=message,
                detector_backend=storage['detector_backend']
            ) # sending result


    # verifing faces
    @bot.message_handler(func=lambda message: message.text == "Verify faces🤓🥸")
    @session_waiter(bot, storage)
    def get_verify_base_image(message: Message):
        sent_message = bot.send_message(message.from_user.id, 'Send photo with base face')
        bot.register_next_step_handler(sent_message, get_verify_viryfing_image)
    

    @session_waiter(bot, storage)
    def get_verify_viryfing_image(message: Message):
        if message.content_type != 'photo':
            bot.send_message(message.from_user.id, "It isn't photo...")
        else:
            image_base = get_image_from_message(bot, message)
            storage['image_base'] = image_base
            bot.send_message(message.from_user.id, "Got image")
            sent_message = bot.send_message(message.from_user.id, "Send photo with verifing face")
            bot.register_next_step_handler(sent_message, verify_faces)
    

    @session_waiter(bot, storage)
    def verify_faces(message: Message):
        if message.content_type != 'photo': # exception
            bot.send_message(message.from_user.id, "It isn't photo...")
        else:
            telegram_verify_faces_functional(
                bot=bot, 
                message=message, 
                image_base=storage['image_base'], 
                detector_backend=storage['detector_backend'], 
                model_name=storage['deepface_analyze']
            ) # sending result
        storage['image_base'] = None # clearing
    

    # analizing face
    @bot.message_handler(func=lambda message: message.text == "Analyze face☹️😀")
    @session_waiter(bot, storage)
    def get_analyze_face(message: Message):
        sent_message = bot.send_message(message.from_user.id, 'Send photo with analized face')
        bot.register_next_step_handler(sent_message, analyze_face)


    @session_waiter(bot, storage)
    def analyze_face(message: Message):
        if message.content_type != 'photo': # exception
            bot.send_message(message.from_user.id, "It isn't photo...")
        else:
            telegram_analyze_face_functional(
                bot=bot, 
                message=message, 
                detector_backend=storage['detector_backend']
            ) # sending results
    

    # object detection
    @bot.message_handler(func=lambda message: message.text == "Object detection🕵️")
    @session_waiter(bot, storage)
    def get_object_detection(message: Message):
        sent_message = bot.send_message(message.from_user.id, 'Send photo with objects')
        bot.register_next_step_handler(sent_message, object_detection)


    @session_waiter(bot, storage)
    def object_detection(message: Message):
        if message.content_type != 'photo': # exception
            bot.send_message(message.from_user.id, "It isn't photo...")
        else:
            telegram_object_detection_functional(
                bot=bot,
                message=message, 
                object_detection_yolo=storage['object_detection_yolo']
            ) # sending results
    

    # segmentation
    @bot.message_handler(func=lambda message: message.text == 'Segmentation✒️')
    @session_waiter(bot, storage)
    def get_segmentation(message: Message):
        sent_message = bot.send_message(message.from_user.id, 'Send photo with objects')
        bot.register_next_step_handler(sent_message, segmentation)


    @session_waiter(bot, storage)
    def segmentation(message: Message):
        if message.content_type != 'photo': # exception
            bot.send_message(message.from_user.id, "It isn't photo...")
        else:
            telegram_segmentation_functional(
                bot=bot,
                message=message,
                segmentation_yolo=storage['segmentation_yolo']
            ) # sending results
    

    # pose people
    @bot.message_handler(func=lambda message: message.text == 'Pose people🧑‍🤝‍🧑')
    @session_waiter(bot, storage)
    def get_pose(message: Message):
        sent_message = bot.send_message(message.from_user.id, 'Send photo with people')
        bot.register_next_step_handler(sent_message, pose)
    

    def pose(message: Message):
        if message.content_type != 'photo': # exception
            bot.send_message(message.from_user.id, "It isn't photo...")
        else:
            telegram_pose_functional(
                bot=bot,
                message=message,
                pose_yolo=storage['pose_yolo']
            ) # sending_results
    

    # settings menu
    @bot.message_handler(func=lambda message: message.text == 'Settings⚙️')
    @session_waiter(bot, storage)
    def choose_settings(message: Message):
        bot.send_message(message.from_user.id, "Choose: ", reply_markup=settings_menu)


    # get info
    @bot.callback_query_handler(func=lambda callback_query: callback_query.data == "Info")
    @session_waiter(bot, storage)
    def send_info(callback_query: CallbackQuery):
        result_msg =  "Info:\n\n"
        result_msg += 'The bot provides an opportunity to work with people faces through a neural network using the libraries YOLOv5, TensorFlow, Keras, Deepface, OpenCV2. Recognition accuracy is ~97%. <a href="https://github.com/CsgoBotTG/DeepFaceTelegramBot"><b>Github</b></a>\n\n'
        result_msg += "Characteristics:\n"
        result_msg += f"\t\t\t[+] <b>Model</b>: {storage['deepface_analyze']}\n"
        result_msg += f"\t\t\t[+] <b>Detector Backend</b>: {storage['detector_backend']}\n"
        result_msg += f"\t\t\t[+] <b>Object Detection Model</b>: {storage['object_detection_yolo']}\n"
        result_msg += f"\t\t\t[+] <b>Segmentation Model</b>: {storage['segmentation_yolo']}\n"
        result_msg += f"\t\t\t[+] <b>Pose Model</b>: {storage['pose_yolo']}"
        bot.send_message(callback_query.from_user.id, result_msg, parse_mode='HTML')


    # choose detector backend
    @bot.callback_query_handler(func=lambda callback_query: callback_query.data == "Detector backend")
    @session_waiter(bot, storage)
    def choose_detector_backend(callback_query: CallbackQuery):
        bot.send_message(callback_query.from_user.id, 'Choose: ', reply_markup=detector_backend_menu)


    @bot.callback_query_handler(func=lambda callback_query: callback_query.data in detector_backends_models)
    @session_waiter(bot, storage)
    def change_detector_backend(callback_query: CallbackQuery):
        storage['detector_backend'] = callback_query.data
        bot.send_message(callback_query.from_user.id, f"Choosed Detector backend: {storage['detector_backend']}")


    # choose analyze model
    @bot.callback_query_handler(func=lambda callback_query: callback_query.data == "Model Neural Network for analyze face")
    @session_waiter(bot, storage)
    def choose_deepface_analyze(callback_query: CallbackQuery):
        bot.send_message(callback_query.from_user.id, 'Choose: ', reply_markup=deepface_analyze_menu)


    @bot.callback_query_handler(func=lambda callback_query: callback_query.data in deepface_analyze_models)
    @session_waiter(bot, storage)
    def change_deepface_analyze(callback_query: CallbackQuery):
        storage['deepface_analyze'] = callback_query.data
        bot.send_message(callback_query.from_user.id, f"Choosed Deepface Analyze Model: {storage['deepface_analyze']}")


    # choose object detection
    @bot.callback_query_handler(func=lambda callback_query: callback_query.data == "Object Detection Yolo Model")
    @session_waiter(bot, storage)
    def choose_object_detection_yolo(callback_query: CallbackQuery):
        bot.send_message(callback_query.from_user.id, 'Choose: ', reply_markup=object_detection_yolo_menu)


    @bot.callback_query_handler(func=lambda callback_query: callback_query.data in object_detection_yolo_models)
    @session_waiter(bot, storage)
    def change_object_detection_yolo(callback_query: CallbackQuery):
        storage['object_detection_yolo'] = callback_query.data
        bot.send_message(callback_query.from_user.id, f"Choosed Object Detection Model: {storage['object_detection_yolo']}")
    

    # choose segmentation
    @bot.callback_query_handler(func=lambda callback_query: callback_query.data == "Segmentation Yolo Model")
    @session_waiter(bot, storage)
    def choose_segmentation_yolo(callback_query: CallbackQuery):
        bot.send_message(callback_query.from_user.id, 'Choose: ', reply_markup=segmentation_yolo_menu)


    @bot.callback_query_handler(func=lambda callback_query: callback_query.data in segmentation_yolo_models)
    @session_waiter(bot, storage)
    def change_segmentation_yolo(callback_query: CallbackQuery):
        storage['segmentation_yolo'] = callback_query.data
        bot.send_message(callback_query.from_user.id, f"Choosed Segmentation Yolo Model: {storage['segmentation_yolo']}")
    

    # choose pose
    @bot.callback_query_handler(func=lambda callback_query: callback_query.data == "Pose Yolo Model")
    @session_waiter(bot, storage)
    def choose_pose_yolo(callback_query: CallbackQuery):
        bot.send_message(callback_query.from_user.id, 'Choose: ', reply_markup=pose_yolo_menu)


    @bot.callback_query_handler(func=lambda callback_query: callback_query.data in pose_yolo_models)
    @session_waiter(bot, storage)
    def change_pose_yolo(callback_query: CallbackQuery):
        storage['pose_yolo'] = callback_query.data
        bot.send_message(callback_query.from_user.id, f"Choosed Pose Yolo Model: {storage['pose_yolo']}")
    

    # finish him!
    @bot.message_handler(func=lambda message: message.text == "Finish Bot👊")
    @session_waiter(bot, storage)
    def exit(message: Message):
        bot.send_message(message.from_user.id, "Exiting...")
        exit(print('Exiting...\n\n'))


    # starting bot
    def main():
        if to_log:
            info_bot = json.loads(bot.get_me().to_json())
            first_name = info_bot['first_name']
            username = info_bot['username']
            print(f"Starting bot {first_name} with token {token} (pyTelegramBotApi). https://t.me/{username} | @{username}")

        bot.polling()

    main()


if __name__ == '__main__':
    start_bot("6637485467:AAFmS9mSSgTQDf8ZrbQQPapJ4neoCAPzBoo")