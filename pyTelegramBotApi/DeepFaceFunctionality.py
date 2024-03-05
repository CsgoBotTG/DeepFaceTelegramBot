import numpy as np

from deepface import DeepFace


def faces_in_photo(
        image: np.ndarray, 
        detector_backend: str = 'opencv'
    ) -> dict:
    """
    Find faces from photo
    
    :param image: np.ndarray. Image where you find faces
    :param detector_backend: str. Used detector_backend

    :return: dict. Face objects
    """

    face_objs = DeepFace.extract_faces(image, detector_backend=detector_backend, enforce_detection=False)

    return face_objs


def verify_in_photo(
        image1: np.ndarray, 
        image2: np.ndarray, 
        detector_backend: str = 'opencv', 
        model_name: str = 'VGG-Face'
    ) -> dict:
    """
    Verify 2 face

    :param image1: np.ndarray. Base face
    :param image2: np.ndarray. Verifing face
    :param detector_backend: str. Used detector_backend
    :param model_name: str. Used model deepface analyzed model
    """

    result = DeepFace.verify(image1, image2, detector_backend=detector_backend, model_name=model_name, enforce_detection=False)

    return result


def analyze_face_in_photo(
        image: np.ndarray, 
        detector_backend='opencv'
    ) -> dict:
    """
    Analyze face: emotions, age, gender, race

    :param image: np.ndarray. Image with face
    :param detector_backend: str. Used detector_backend

    :return: dict. Analyzed face in json
    """

    analyze = DeepFace.analyze(image, actions = ['age', 'gender', 'race', 'emotion'], detector_backend=detector_backend, enforce_detection=False)

    return analyze