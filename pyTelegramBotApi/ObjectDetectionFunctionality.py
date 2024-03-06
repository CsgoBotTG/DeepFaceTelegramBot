import cv2
import random
import cvzone
import numpy as np
import ultralytics

from TelegramBotNames import models_path

def object_detect_with_highligth(
        image: np.ndarray, 
        obj_detect_model: str = "yolov8x.pt"
    ) -> None:
    """
    Object detection from image

    :param image: np.ndarray. Image highligthed
    :param obj_detect_model: str. Path to model of object detection yolov8
    :param models_path: str. path to model with yolov8 object_detection

    :return: result (json) and highligthed image
    """

    model = ultralytics.YOLO(models_path + obj_detect_model)
    result = model.predict(image)

    for r in result:
        boxes = r.boxes
        for box in boxes:
            conf = np.ceil(box.conf[0]*100)/100

            x1, y1, x2, y2 = list(map(int, box.xyxy[0]))
            w, h = x2 - x1, y2 - y1

            curr_class = model.names[int(box.cls[0])]

            if conf > 0.55:
                cvzone.putTextRect(image, f'{curr_class} | {conf}', (max(0, x1), max(35, y1)), scale=0.75, thickness=1, offset=3)
                cvzone.cornerRect(image, (x1, y1, w, h), l=9)

    return result, image


def segmentation_detect_with_higligth(
        image: np.ndarray,
        segmentation_model: str = "yolov8n-seg.pt"
    ):
    """
    Segmentation from image

    :param image: np.ndarray. Image highligthed
    :param segmentation_model: str. Path to model of segmentation yolov8
    :param models_path: str. Path to dir with models

    :return: results (json) and image highlighted
    """
    image_highlighted = image.copy()

    model = ultralytics.YOLO(models_path + segmentation_model)
    yolo_classes = list(model.names.values())
    classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]

    conf = 0.5

    results = model.predict(image_highlighted, conf=conf)
    colors = [random.choices(range(256), k=3) for _ in classes_ids]
    
    for result in results:
        for mask, box in zip(result.masks.xy, result.boxes):
            points = np.int32([mask])
            color_number = classes_ids.index(int(box.cls[0]))
            cv2.polylines(image_highlighted, points, True, colors[color_number], 3)


    return results, image_highlighted