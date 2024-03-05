import cv2
import numpy as np
import requests

from aiogram import Bot
from aiogram.types import Message
from aiogram.fsm.context import FSMContext
from aiogram.types import InputFile


def session_waiter(bot: Bot, storage: dict):
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


def session_waiter_with_state(bot: Bot, storage: dict):
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


async def get_image_from_message(bot: Bot, message: Message):
    file_id = message.photo[-1].file_id
    file_info = await bot.get_file(file_id)
    file_path = file_info.file_path
    file_bytes = await bot.download_file(file_path)
    file_bytes_arr = np.frombuffer(file_bytes.read(), np.uint8)
    file = cv2.imdecode(file_bytes_arr, cv2.IMREAD_COLOR)

    return file


def send_image(bot: Bot, message: Message, image: np.ndarray, caption: str = None):
    image = cv2.imencode('.jpg', image)[1].tostring()
    
    data = {'chat_id': message.from_user.id}
    if not caption is None:
        data['caption'] = caption
    url = f"https://api.telegram.org/bot{bot.token}/sendPhoto"

    result = requests.post(url, data=data, files={'photo': image})
    return result.json()