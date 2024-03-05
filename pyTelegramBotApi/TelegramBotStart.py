import telebot
import TelegramBotFunctional


from time import sleep


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
object_detection_yolo_models = [
    'yolov8n.pt',
    'yolov8s.pt',
    'yolov8m.pt',
    'yolov8l.pt',
    'yolov8x.pt'
]


model = models_deepface[-1]
detector_backend = detector_backends[-1]
object_detection_yolo = object_detection_yolo_models[-1]

session = False
first_message = True
image_base = None

def get_address_bot(token: str = None) -> dict:
    if token is None:
        return token
    
    bot = telebot.TeleBot(token=token)
    result = bot.get_me().to_dict()

    return result

def TelegramBotStart(token: str = None) -> None:
    if token is None:
        raise 'No token'

    bot = telebot.TeleBot(token=token)
    
    markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
    buttons = (
        telebot.types.KeyboardButton("Find faces in a photo🔎"), 
        telebot.types.KeyboardButton("Verify faces🤓🥸"),
        telebot.types.KeyboardButton("Analyze face☹️😀"),
        telebot.types.KeyboardButton("Object detection🕵️"),
        telebot.types.KeyboardButton("Settings⚙️"),
    )
    markup.add(buttons[0], buttons[1], buttons[2])
    markup.add(buttons[3])
    markup.add(buttons[4])

    settings_markup = telebot.types.InlineKeyboardMarkup()
    buttons_settings = (
        "Detector backend",
        "Model Neural Network",
        "Object Detection Yolo Model",
        "Info",
    )
    [settings_markup.add(
        telebot.types.InlineKeyboardButton(text=button, 
                                           callback_data=button)
                                ) for button in buttons_settings]
    
    detector_backend_markup = telebot.types.InlineKeyboardMarkup()
    [detector_backend_markup.add(
        telebot.types.InlineKeyboardButton(text=detector_backend, 
                                           callback_data=detector_backend)
                                ) for detector_backend in detector_backends]
    
    model_deepface_markup = telebot.types.InlineKeyboardMarkup()
    [model_deepface_markup.add(
        telebot.types.InlineKeyboardButton(text=model, 
                                           callback_data=model)
                                ) for model in models_deepface]
    
    object_detection_yolo_markup = telebot.types.InlineKeyboardMarkup()
    [object_detection_yolo_markup.add(
        telebot.types.InlineKeyboardButton(text=model, 
                                           callback_data=model)
                                ) for model in object_detection_yolo_models]


    @bot.message_handler(commands=['start', 'help'])
    def start(message: telebot.types.Message):
        global first_message, session

        if session:
            bot.send_message(message.chat.id, "Sorry i couln't do it now. Wait...")
        else:
            session = True
            if first_message:
                bot.send_video(message.chat.id, 'https://media1.tenor.com/m/5hKPyupKGWMAAAAC/robot-hello.gif')
                bot.send_message(message.chat.id, f"Hello! I'm BOT that working on YOLOv5, Deepface and Tensorflow!")

                first_message = False
                
            bot.send_message(message.chat.id, f"My functionality are: " + \
                            "\n1. DeepFace\n\t\t\t\t1. Find faces in a photo;\n\t\t\t\t2. Verify faces several photos;" + \
                            "\n\t\t\t\t3. Analyze facial emotions;\n" + \
                            "2. YoloV8\n\t\t\t\t1. Object Detection", 
                            reply_markup=markup)
            
            session = False
    
    @bot.message_handler(content_types=['text'])
    def get_command(message: telebot.types.Message):
        global session
        if session:
            bot.send_message(message.chat.id, "Sorry i couln't do it now. Wait...")
        else:
            session = True
            if message.text == "Find faces in a photo🔎":
                sent_message = bot.send_message(message.chat.id, 'Send photo')
                bot.register_next_step_handler(sent_message, TelegramFindFaces)
            elif message.text == 'Verify faces🤓🥸':
                sent_message = bot.send_message(message.chat.id, 'Send photo with base face')
                bot.register_next_step_handler(sent_message, TelegramVerifyFace1)
            elif message.text == 'Analyze face☹️😀':
                sent_message = bot.send_message(message.chat.id, 'Send photo')
                bot.register_next_step_handler(sent_message, TelegramAnalyzeFace)
            elif message.text == 'Object detection🕵️':
                sent_message = bot.send_message(message.chat.id, 'Send photo')
                bot.register_next_step_handler(sent_message, TelegramObjectDetection)
            elif message.text == 'Settings⚙️':
                sent_message = bot.send_message(message.chat.id, "Choose what's you wanna change", reply_markup=settings_markup)
            else:
                bot.send_message(message.chat.id, "I didn't get it...")
            session = False


    # find faces
    def TelegramFindFaces(message: telebot.types.Message):
        global session
        if session:
            bot.send_message(message.chat.id, "Sorry i couln't do it now. Wait...")
        else:
            session = True
            if message.content_type != 'photo':
                bot.send_message(message.chat.id, "It isn't photo...")
            else:
                TelegramBotFunctional.TelegramFindFacesFunctional(bot, message, detector_backend)
            session = False
    

    # verify faces
    def TelegramVerifyFace1(message):
        global image_base, session

        if session:
            bot.send_message(message.chat.id, "Sorry i couln't do it now. Wait...")
        else:
            session = True
            if message.content_type != 'photo':
                bot.send_message(message.chat.id, "It isn't photo...")
            else:
                image_base = TelegramBotFunctional.TelegramVerifyFace1(bot, message)

                sent_messange = bot.send_message(message.chat.id, 'Now send photo with verifing face')
                bot.register_next_step_handler(sent_messange, TelegramVerifyFace2)
            session = False
    
    def TelegramVerifyFace2(message):
        global image_base, session

        if session:
            bot.send_message(message.chat.id, "Sorry i couln't do it now. Wait...")
        else:
            session = True
            if message.content_type != 'photo':
                bot.send_message(message.chat.id, "It isn't photo...")
            else:
                TelegramBotFunctional.TelegramVerifyFace2(bot, message, image_base, detector_backend, model)
            session = False

    
    # analyze face
    def TelegramAnalyzeFace(message: telebot.types.Message):
        global session
        if session:
            bot.send_message(message.chat.id, "Sorry i couln't do it now. Wait...")
        else:
            session = True
            if message.content_type != 'photo':
                bot.send_message(message.chat.id, "It isn't photo...")
            else:
                TelegramBotFunctional.TelegramAnalyzeFace(bot, message, markup, detector_backend)
            session = False

    # object detection
    def TelegramObjectDetection(message):
        global session
        if session:
            bot.send_message(message.chat.id, "Sorry i couln't do it now. Wait...")
        else:
            session = True
            if message.content_type != 'photo':
                bot.send_message(message.chat.id, "It isn't photo...")
            else:
                TelegramBotFunctional.TelegramObjectDetection(bot, message, object_detection_yolo)

            session = False
    

    # settings
    @bot.callback_query_handler(func=lambda call: call.data == 'Info')
    def send_info(call):
        global session
        if session:
            bot.send_message(call.message.chat.id, "Sorry i couln't do it now. Wait...")
        else:
            session = True
            
            result_msg =  "The bot provides an opportunity to work with neural networks using the libraries YOLOv5, TensorFlow, Keras, Deepface, OpenCV2. Recognition accuracy is ~97%\n\n"
            result_msg += f'Characteristics:\n'
            result_msg += f'\t\t\t[+] Model: {model}\n'
            result_msg += f'\t\t\t[+] Detector_backend: {detector_backend}\n'
            result_msg += f'\t\t\t[+] Object Detection Model: {object_detection_yolo}'

            bot.send_message(call.message.chat.id, result_msg, reply_markup=markup)
            session = False
    
    # model obj detect yolo name
    @bot.callback_query_handler(func=lambda call: call.data == "Object Detection Yolo Model")
    def change_obj_detection_yolo_model(call):
        global session
        if session:
            bot.send_message(call.message.chat.id, "Sorry i couln't do it now. Wait...")
        else:
            session = True
            bot.send_message(call.message.chat.id, 'Choose model you want', reply_markup=object_detection_yolo_markup)
            session = False
    
    @bot.callback_query_handler(func=lambda call: call.data in object_detection_yolo_models)
    def change_obj_detection_yolo_n(call):
        global session
        if session:
            bot.send_message(call.message.chat.id, "Sorry i couln't do it now. Wait...")
        else:
            session = True
            bot.send_message(call.message.chat.id, 'Changing...')

            global object_detection_yolo
            object_detection_yolo = call.data

            bot.send_message(call.message.chat.id, f'Changed on {call.data}', reply_markup=markup)
            session = False
    
    
    # Model name
    @bot.callback_query_handler(func=lambda call: call.data == 'Model Neural Network')
    def change_model_nn(call):
        global session
        if session:
            bot.send_message(call.message.chat.id, "Sorry i couln't do it now. Wait...")
        else:
            session = True
            bot.send_message(call.message.chat.id, 'Choose model you want', reply_markup=model_deepface_markup)
            session = False
    
    @bot.callback_query_handler(func=lambda call: call.data in models_deepface)
    def change_detector_backend_opencv(call):
        global session
        if session:
            bot.send_message(call.message.chat.id, "Sorry i couln't do it now. Wait...")
        else:
            session = True
            bot.send_message(call.message.chat.id, 'Changing...')

            global model
            model = call.data

            bot.send_message(call.message.chat.id, f'Changed on {call.data}', reply_markup=markup)
            session = False
    

    # detector_backend
    @bot.callback_query_handler(func=lambda call: call.data == 'Detector backend')
    def change_detector_backend(call):
        global session
        if session:
            bot.send_message(call.message.chat.id, "Sorry i couln't do it now. Wait...")
        else:
            session = True
            bot.send_message(call.message.chat.id, 'Choose detector backend you want', reply_markup=detector_backend_markup)
            session = False
    
    @bot.callback_query_handler(func=lambda call: call.data in detector_backends)
    def change_detector_backend_opencv(call):
        global session
        if session:
            bot.send_message(call.message.chat.id, "Sorry i couln't do it now. Wait...")
        else:
            session = True
            bot.send_message(call.message.chat.id, 'Changing...')

            global detector_backend
            detector_backend = call.data

            bot.send_message(call.message.chat.id, f'Changed on {call.data}', reply_markup=markup)
            session = False
    
    bot.infinity_polling(none_stop=True)



if __name__ == '__main__':
    print(TelegramBotStart(token='6637485467:AAFmS9mSSgTQDf8ZrbQQPapJ4neoCAPzBoo'))
