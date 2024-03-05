import cv2
import numpy as np

from deepface import DeepFace


def FacesInPhoto(image: np.ndarray = None, detector_backend='opencv') -> dict:
    if image is None:
        raise 'No Image'
    
    face_objs = DeepFace.extract_faces(image, detector_backend=detector_backend, enforce_detection=False)
    
    return face_objs


def VerifyInPhotos(image1: np.ndarray = None, image2: np.ndarray = None, detector_backend='opencv', model_name='VGG-Face') -> dict:
    if image1 is None or image2 is None:
        raise 'No Image'
    
    result = DeepFace.verify(image1, image2, detector_backend=detector_backend, model_name=model_name, enforce_detection=False)

    return result


def AnalyzeFaceEmotionInPhoto(image: np.ndarray = None, detector_backend='opencv') -> dict:
    if image is None:
        raise 'No Image'
    
    analyze = DeepFace.analyze(image, actions = ['age', 'gender', 'race', 'emotion'], detector_backend=detector_backend, enforce_detection=False)

    return analyze



if __name__ == '__main__':
    # Analyze Emotions
    '''
    image = cv2.imread('images/Emotion.png')
    analyze = AnalyzeFaceEmotionInPhoto(image)[0]

    gender = 'Woman'
    if analyze.get('gender').get('Man') > analyze.get('gender').get('Woman'):
        gender = 'Man'

    #print(json.dumps(analyze, indent=2))pylin
    print('\n\n')
    print(f'[+] Age: {analyze.get("age")}')
    print(f'[+] Gender: {gender}')
    print(f'[+] Race:')

    for k, v in analyze.get('race').items():
        print(f'\t{k} - {round(v, 2)}%')
    
    print(f'[+] Emotions:')

    for k, v in analyze.get('emotion').items():
        print(f'\t{k} - {round(v, 2)}%')

    face_region = analyze['region']
    print(face_region)
    print(face_region['x'], face_region['x']+face_region['w'], face_region['y'], face_region['y']+face_region['h'])
    face = image[face_region['y']:face_region['y']+face_region['h'], face_region['x']:face_region['x']+face_region['w']]

    cv2.rectangle(image, (face_region['x'], face_region['y']), (face_region['x']+face_region['w'], face_region['y']+face_region['h']), (255, 69, 69), 4)
    cv2.imwrite('images/EmotionResult.png', image)

    cv2.imshow('face', face)
    cv2.waitKey(10000)
    '''


    # FacesInPhoto
    '''
    image = cv2.imread('images/Volleyball.jpg')
    result = FacesInPhoto(image)

    for face in result:
        area = face['facial_area']
        cv2.rectangle(image, (area['x'], area['y']), (area['x'] + area['w'], area['y'] + area['h']), (255, 69, 69), 4)
    
    cv2.imwrite('images/VolleyballResult.jpg', image)
    '''


    # SameFacesInPhotos
    '''
    image1 = cv2.imread('images/Harry1.jpg')
    image2 = cv2.imread('images/Harry2.jpg')
    #image2 = cv2.imread('images/Emotion.png')

    result = VerifyInPhotos(image1, image2)
    #print(result)

    cv2.rectangle(image1, (result['facial_areas']['img1']['x'], result['facial_areas']['img1']['y']), (result['facial_areas']['img1']['x'] + result['facial_areas']['img1']['w'], result['facial_areas']['img1']['y'] + result['facial_areas']['img1']['h']), (255, 69, 69), 3)
    cv2.rectangle(image2, (result['facial_areas']['img2']['x'], result['facial_areas']['img2']['y']), (result['facial_areas']['img2']['x'] + result['facial_areas']['img2']['w'], result['facial_areas']['img2']['y'] + result['facial_areas']['img2']['h']), (255, 69, 69), 3)


    if image1.shape[0] * image1.shape[1] > image2.shape[0] * image2.shape[1]:
        image1 = cv2.resize(image1, image2.shape[:2][::-1])
    else:
        image_verify = cv2.resize(image2, image1.shape[:2][::-1])

    vis = np.concatenate((image1, image2), axis=1)
    cv2.imwrite('images/HarryResult.jpg', vis)
    cv2.imshow('vis', vis)
    cv2.waitKey(10000)
    '''