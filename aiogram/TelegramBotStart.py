import asyncio

from TelegramBotNames import *
from TelegramBotFunctions import *
from TelegramBotFunctionsHelper import *

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.enums import ParseMode
from aiogram.types import Message, CallbackQuery
from aiogram.fsm.context import FSMContext


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
    bot = Bot(token)
    dp = Dispatcher()
    storage = {
        'deepface_analyze': deepface_analyze_models[-1],
        'detector_backend': detector_backends_models[-1],
        'object_detection_yolo': object_detection_yolo_models[-1],
        'segmentation_yolo': segmentation_yolo_models[-1],
        'pose_yolo': pose_yolo_models[-1],
        'first_message': True,
        'session': False,
    }

    # start/help menu
    @dp.message(Command(commands=['start', 'help']))
    @session_waiter(bot, storage)
    async def start(message: Message):
        if storage['first_message']:
            await bot.send_animation(message.from_user.id, 'https://media1.tenor.com/m/5hKPyupKGWMAAAAC/robot-hello.gif')
            await bot.send_message(message.from_user.id, f"Hello, {message.from_user.first_name}! I'm BOT that working on YOLOv5, Deepface, Tensorflow and AIoGram!")

            storage['first_message'] = False

        await bot.send_message(message.from_user.id, 
                                "My functionality are: " + \
                                "\n1. DeepFace\n\t\t\t\t1. Find faces in a photo;\n\t\t\t\t" + \
                                "2. Verify faces several photos;" + \
                                "\n\t\t\t\t3. Analyze facial emotions;\n" + \
                                "2. YoloV8\n\t\t\t\t1. Object Detection;\n\t\t\t\t" + \
                                "2. Segmentation;\n\t\t\t\t" + \
                                "3. Pose people;", 
                                reply_markup=start_menu)


    # find faces
    @dp.message(F.text == 'Find faces in a photo🔎')
    @session_waiter_with_state(bot, storage)
    async def get_find_faces(message: Message, state: FSMContext):
        await state.set_state(FindFacesForm.image) # set state to get image
        await bot.send_message(message.from_user.id, 'Send photo where you wanna highligth faces')


    @dp.message(FindFacesForm.image)
    @session_waiter_with_state(bot, storage)
    async def find_faces(message: Message, state: FSMContext):
        if message.content_type != 'photo': # exceptions
            await bot.send_message(message.from_user.id, "It isn't photo...")
        else:
            await telegram_find_face_functional(
                bot=bot, 
                message=message,
                detector_backend=storage['detector_backend']
            ) # sending result
        await state.clear()


    # verifing faces
    @dp.message(F.text == "Verify faces🤓🥸")
    @session_waiter_with_state(bot, storage)
    async def get_verify_base_image(message: Message, state: FSMContext):
        await state.set_state(VerifyFaceForm.image_base) # set state to get image
        await bot.send_message(message.from_user.id, 'Send photo with base face')


    @dp.message(VerifyFaceForm.image_base)
    @session_waiter_with_state(bot, storage)
    async def get_verify_base_image2(message: Message, state: FSMContext):
        if message.content_type != 'photo': # exception
            await bot.send_message(message.from_user.id, "It isn't photo...")
            await state.clear()
        else:
            image_base = await get_image_from_message(bot, message) # get image
            await bot.send_message(message.from_user.id, "Got image")
            await state.update_data(image_base=image_base) # update data
            await bot.send_message(message.from_user.id, "Send photo with verifing face")
            await state.set_state(VerifyFaceForm.image_verify) # set state to get second image


    @dp.message(VerifyFaceForm.image_verify)
    @session_waiter_with_state(bot, storage)
    async def verify_base_image(message: Message, state: FSMContext):
        if message.content_type != 'photo': # exception
            await bot.send_message(message.from_user.id, "It isn't photo...")
        else:
            data = await state.get_data() # get data
            await telegram_verify_faces_functional(
                bot=bot, 
                message=message, 
                image_base=data['image_base'], 
                detector_backend=storage['detector_backend'], 
                model_name=storage['deepface_analyze']
            ) # sending result
        await state.clear()


    # analizing face
    @dp.message(F.text == "Analyze face☹️😀")
    @session_waiter_with_state(bot, storage)
    async def get_analyze_face(message: Message, state: FSMContext):
        await state.set_state(AnalyzeFaceForm.image) # set state to get image
        await bot.send_message(message.from_user.id, 'Send photo with analized face')


    @dp.message(AnalyzeFaceForm.image)
    @session_waiter_with_state(bot, storage)
    async def analyze_face(message: Message, state: FSMContext):
        if message.content_type != 'photo': # exception
            await bot.send_message(message.from_user.id, "It isn't photo...")
        else:
            await telegram_analyze_face_functional(
                bot=bot, 
                message=message, 
                detector_backend=storage['detector_backend']
            ) # sending results
        await state.clear()


    # object detection
    @dp.message(F.text == "Object detection🕵️")
    @session_waiter_with_state(bot, storage)
    async def get_object_detection(message: Message, state: FSMContext):
        await state.set_state(ObjectDetectionYoloForm.image) # set state to get image
        await bot.send_message(message.from_user.id, 'Send photo with objects')


    @dp.message(ObjectDetectionYoloForm.image)
    @session_waiter_with_state(bot, storage)
    async def object_detection(message: Message, state: FSMContext):
        if message.content_type != 'photo': # exception
            await bot.send_message(message.from_user.id, "It isn't photo...")
        else:
            await telegram_object_detection_functional(
                bot=bot,
                message=message, 
                object_detection_yolo=storage['object_detection_yolo']
            ) # sending results
        await state.clear()


    # segmentation
    @dp.message(F.text == 'Segmentation✒️')
    @session_waiter_with_state(bot, storage)
    async def get_segmentation(message: Message, state: FSMContext):
        await state.set_state(SegmentationYoloForm.image)
        await bot.send_message(message.from_user.id, 'Send photo with objects')


    @dp.message(SegmentationYoloForm.image)
    @session_waiter_with_state(bot, storage)
    async def segmentation(message: Message, state: FSMContext):
        if message.content_type != 'photo': # exception
            await bot.send_message(message.from_user.id, "It isn't photo...")
        else:
            await telegram_segmentation_functional(
                bot=bot,
                message=message,
                segmentation_yolo=storage['segmentation_yolo']
            ) # sending results
        await state.clear()
    

    # pose people
    @dp.message(F.text == 'Pose people🧑‍🤝‍🧑')
    @session_waiter_with_state(bot, storage)
    async def get_pose(message: Message, state: FSMContext):
        await state.set_state(PoseYoloForm.image)
        await bot.send_message(message.from_user.id, 'Send photo with people')
    

    @dp.message(PoseYoloForm.image)
    @session_waiter_with_state(bot, storage)
    async def pose(message: Message, state: FSMContext):
        if message.content_type != 'photo': # exception
            await bot.send_message(message.from_user.id, "It isn't photo...")
        else:
            await telegram_pose_functional(
                bot=bot,
                message=message,
                pose_yolo=storage['pose_yolo']
            ) # sending results
        await state.clear()


    # settings menu
    @dp.message(F.text == 'Settings⚙️')
    @session_waiter(bot, storage)
    async def choose_settings(message: Message):
        await bot.send_message(message.from_user.id, "Choose: ", reply_markup=settings_menu)


    # get info
    @dp.callback_query(F.data == "Info")
    @session_waiter(bot, storage)
    async def send_info(callback_query: CallbackQuery):
        result_msg =  "Info:\n\n"
        result_msg += 'The bot provides an opportunity to work with people faces through a neural network using the libraries YOLOv5, TensorFlow, Keras, Deepface, OpenCV2. Recognition accuracy is ~97%. <a href="https://github.com/CsgoBotTG/DeepFaceTelegramBot"><b>Github</b></a>\n\n'
        result_msg += "Characteristics:\n"
        result_msg += f"\t\t\t[+] <b>Model</b>: {storage['deepface_analyze']}\n"
        result_msg += f"\t\t\t[+] <b>Detector Backend</b>: {storage['detector_backend']}\n"
        result_msg += f"\t\t\t[+] <b>Object Detection Model</b>: {storage['object_detection_yolo']}\n"
        result_msg += f"\t\t\t[+] <b>Segmentation Model</b>: {storage['segmentation_yolo']}\n"
        result_msg += f"\t\t\t[+] <b>Pose Model</b>: {storage['pose_yolo']}"
        await callback_query.message.edit_text(result_msg, parse_mode=ParseMode.HTML)


    # choose detector backend
    @dp.callback_query(F.data == "Detector backend")
    @session_waiter(bot, storage)
    async def choose_detector_backend(callback_query: CallbackQuery):
        await callback_query.message.edit_text("Choose: ", reply_markup=detector_backend_menu)


    @dp.callback_query(F.data.in_(detector_backends_models))
    @session_waiter(bot, storage)
    async def change_detector_backend(callback_query: CallbackQuery):
        storage['detector_backend'] = callback_query.data
        await callback_query.message.edit_text(f"Choosed Detector backend: {storage['detector_backend']}")


    # choose analyze model
    @dp.callback_query(F.data == "Model Neural Network for analyze face")
    @session_waiter(bot, storage)
    async def choose_deepface_analyze(callback_query: CallbackQuery):
        await callback_query.message.edit_text("Choose: ", reply_markup=deepface_analyze_menu)


    @dp.callback_query(F.data.in_(deepface_analyze_models))
    @session_waiter(bot, storage)
    async def change_deepface_analyze(callback_query: CallbackQuery):
        storage['deepface_analyze'] = callback_query.data
        await callback_query.message.edit_text(f"Choosed Model Neural Network for analyze face: {storage['deepface_analyze']}")


    # choose object detection
    @dp.callback_query(F.data == "Object Detection Yolo Model")
    @session_waiter(bot, storage)
    async def choose_object_detection_yolo(callback_query: CallbackQuery):
        await callback_query.message.edit_text("Choose: ", reply_markup=object_detection_yolo_menu)


    @dp.callback_query(F.data.in_(object_detection_yolo_models))
    @session_waiter(bot, storage)
    async def change_object_detection_yolo(callback_query: CallbackQuery):
        storage['object_detection_yolo'] = callback_query.data
        await callback_query.message.edit_text(f"Choosed Object Detection Yolo Model: {storage['object_detection_yolo']}")
    

    # choose segmentation
    @dp.callback_query(F.data == "Segmentation Yolo Model")
    @session_waiter(bot, storage)
    async def choose_segmentation_yolo(callback_query: CallbackQuery):
        await callback_query.message.edit_text(F"Choose: ", reply_markup=segmentation_yolo_menu)


    @dp.callback_query(F.data.in_(segmentation_yolo_models))
    @session_waiter(bot, storage)
    async def change_segmentation_yolo(callback_query: CallbackQuery):
        storage['segmentation_yolo'] = callback_query.data
        await callback_query.message.edit_text(f"Choosed Segmentation Yolo Model: {storage['segmentation_yolo']}")
    

    # choose pose
    @dp.callback_query(F.data == "Pose Yolo Model")
    @session_waiter(bot, storage)
    async def choose_pose_yolo(callback_query: CallbackQuery):
        await callback_query.message.edit_text("Choose: ", reply_markup=pose_yolo_menu)


    @dp.callback_query(F.data.in_(pose_yolo_models))
    @session_waiter(bot, storage)
    async def change_pose_yolo(callback_query: CallbackQuery):
        storage['pose_yolo'] = callback_query.data
        await callback_query.message.edit_text(f"Choosed Pose Yolo Model: {storage['pose_yolo']}")
    

    # finish him!
    @dp.message(F.text == 'Finish Bot👊')
    @session_waiter(bot, storage)
    async def exit(message: Message):
        await bot.send_message(message.from_user.id, 'Exiting...')
        print('Exiting...\n\n')
        await dp.stop_polling()


    # starting bot
    async def main():
        await bot.delete_webhook(drop_pending_updates=True)

        if to_log:
            info_bot = await bot.get_me()
            print(f"Starting bot {info_bot.first_name} with token {token} (aiogram). https://t.me/{info_bot.username} | @{info_bot.username}")

        await dp.start_polling(bot)

    asyncio.run(main())