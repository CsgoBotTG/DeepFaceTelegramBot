import cv2
import numpy as np

from aiogram import Bot
from aiogram.types import Message
from aiogram.types.input_file import BufferedInputFile
from aiogram.fsm.context import FSMContext


def session_waiter(
        bot: Bot, 
        storage: dict
    ):
    """
    Waiting for end session

    :param bot: aiogram.Bot. for sending message if session
    :param storage: dict. to get session info

    :return: function. wrapper
    """

    def session_waiter_wrapper(func):
        async def wrapper(message: Message):
            if storage['session']:
                await bot.send_message(message.from_user.id, "Sorry i couln't do it now. Wait...")
            else:
                storage['session'] = True
                await func(message)
                storage['session'] = False
        return wrapper
    return session_waiter_wrapper


def session_waiter_with_state(
        bot: Bot, 
        storage: dict
    ):
    """
    Waiting for end session

    :param bot: aiogram.Bot. for sending message if session
    :param storage: dict. to get session info

    :return: function. wrapper
    """

    def session_waiter_with_state_wrapper(func):
        async def wrapper(message: Message, state: FSMContext):
            if storage['session']:
                await bot.send_message(message.from_user.id, "Sorry i couln't do it now. Wait...")
            else:
                storage['session'] = True
                await func(message, state)
                storage['session'] = False
        return wrapper
    return session_waiter_with_state_wrapper


async def get_image_from_message(
        bot: Bot, 
        message: Message
    ) -> np.ndarray:
    """
    Get image from message

    :param bot: aiogram.Bot. Bot to get image
    :param message: aiogram.types.Messsage. To get image from message

    :return: np.ndarray. Image
    """

    file_id = message.photo[-1].file_id
    file_info = await bot.get_file(file_id)
    file_path = file_info.file_path
    file_bytes = await bot.download_file(file_path)
    file_bytes_arr = np.frombuffer(file_bytes.read(), np.uint8)
    file = cv2.imdecode(file_bytes_arr, cv2.IMREAD_COLOR)

    return file


async def send_image(
        bot: Bot,
        message: Message,
        image: np.ndarray,
        caption: str = None
    ):
    """
    Send image

    :param bot: aiogram.Bot. Bot to get image
    :param message: aiogram.types.Messsage. To get chat id
    :param image: np.ndarray. Image what sent
    :param caption: str | None. String with image sended

    :raturn: None
    """

    _, image_bytes = cv2.imencode('.jpg', image)
    image_to_send = BufferedInputFile(image_bytes.tobytes(), 'image.jpg')

    await bot.send_photo(message.from_user.id, image_to_send, caption=caption)