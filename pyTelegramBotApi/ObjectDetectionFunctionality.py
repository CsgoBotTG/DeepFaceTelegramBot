import cv2
import cvzone
import numpy as np
import ultralytics


def object_detect_with_highligth(image: np.ndarray = None, obj_detect_model: str = "yolov8n"):
    if image is None:
        raise 'No image'

    model = ultralytics.YOLO("weightsYolo/" + obj_detect_model)
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



if __name__ == '__main__':
    image = cv2.imread('images/Town.jpg')
    image_highlithed = object_detect_with_highligth(image, "yolov8n")

    vis = np.concatenate((image, image_highlithed), axis=1)
    cv2.imshow('vis', vis)
    cv2.waitKey(10000)