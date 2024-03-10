import cv2
import numpy as np

from telebot import TeleBot
from telebot.types import Message

def session_waiter(
        bot: TeleBot, 
        storage: dict
    ):
    """
    Waiting for end session

    :param bot: telebot.TeleBot. for sending message if session
    :param storage: dict. to get session info

    :return: function. wrapper
    """

    def session_waiter_wrapper(func):
        def wrapper(message: Message):
            if storage['session']:
                bot.send_message(message.from_user.id, "Sorry i couln't do it now. Wait...")
            else:
                storage['session'] = True
                func(message)
                storage['session'] = False
        return wrapper
    return session_waiter_wrapper


def get_image_from_message(
        bot: TeleBot,
        message: Message
    ) -> np.ndarray:
    """
    Get image from message

    :param bot: telebot.TeleBot. to get file
    :param message: dict. to get file id

    :return: np.ndarray. Image in array
    """

    file_info = bot.get_file(message.photo[len(message.photo)-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    image_bytes_arr = np.fromstring(downloaded_file, np.uint8)
    image = cv2.imdecode(image_bytes_arr, cv2.IMREAD_COLOR)

    return image


def send_image(
        bot: TeleBot,
        message: Message,
        image: np.ndarray, 
        caption: str = None
    ) -> bool:
    """
    Send image

    :param bot: telebot.TeleBot. to send photo
    :param message: dict. to get chat id
    :param image: np.ndarray. for sending photo
    :param caption: str. CAPTION YOU KNOW RIGHT??

    :return: bool. Result
    """

    image_bytes = cv2.imencode('.jpg', image)[1].tostring()
    bot.send_photo(message.from_user.id, image_bytes, caption)

    return True