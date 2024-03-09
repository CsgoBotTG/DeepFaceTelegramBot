from typing import Optional, Tuple

import cv2
import random
import cvzone
import numpy as np
import ultralytics

from TelegramBotNames import models_path

def object_detect_with_highligth(
        image: Optional[np.ndarray], 
        obj_detect_model: Optional[str] = "yolov8x.pt",
    ) -> Tuple[dict, np.ndarray]:
    """
    Object detection from image

    :param image: np.ndarray. Image highligthed
    :param obj_detect_model: str. Path to model of object detection yolov8

    :return: result (dict) and highligthed image
    """

    image_highlighted = image.copy()

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

    return result, image_highlighted


def segmentation_detect_with_higligth(
        image: Optional[np.ndarray],
        segmentation_model: Optional[str] = "yolov8x-seg.pt",
    ) -> Tuple[dict, np.ndarray]:
    """
    Segmentation from image

    :param image: np.ndarray. Image highligthed
    :param segmentation_model: str. Path to model of segmentation yolov8

    :return: results (dict) and image highlighted
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


def pose_detect_with_highligth(
        image: Optional[np.ndarray],
        pose_model: Optional[np.ndarray] = "yolov8x-pose.pt"
    ) -> Tuple[dict, np.ndarray]:
    """
    Pose people from image

    :param image: np.ndarray. Image highligthed
    :param pose_model: str. Path to model of segmentation yolov8
    
    :return: results (dict) and image highlighted
    """

    image_highlighted = image.copy()

    model = ultralytics.YOLO(models_path + pose_model)
    results = model.predict(image)

    pairs = (
        (4, 2),
        (2, 0),
        (0, 1),
        (1, 3),
        (10, 8), 
        (8, 6),
        (6, 5),
        (5, 7),
        (7, 9),
        (6, 12),
        (12, 11),
        (5, 11),
        (12, 14),
        (14, 16),
        (11, 13),
        (13, 15),
    )

    for result in results:
        for result_keypoints in result.keypoints.xy.cpu().numpy():
            for keypoint in result_keypoints:
                if sum(list(keypoint)) > 0:
                    cv2.circle(image_highlighted, list(map(int, list(keypoint))), 5, (69, 255, 69), 5)
            
            for pair in pairs:
                point1 = result_keypoints[pair[0]]
                point2 = result_keypoints[pair[1]]
                if sum(list(point1)) > 0 and sum(list(point2)) > 0:
                    cv2.line(image_highlighted, list(map(int, list(point1))), list(map(int, list(point2))), (255, 69, 69), 5)

    return results, image_highlighted