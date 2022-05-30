import logging
import cv2
import dlib
import json
import numpy as np
from fer import FER

def shape_to_np_right(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(36, 42):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def shape_to_np_left(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(42, 48):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

detector = FER(mtcnn=True)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("face_shape.dat")

def video_processing(file_path):
    video_file = file_path.partition("/")[2]
    video_name = video_file.split(".")[0]

    pos_eye_left = []
    pos_eye_right = []
    results = []
    emotions = []
    n = 1
    i = 1

    cap = cv2.VideoCapture(file_path)

    while i == 1:
        try:
            _, frame = cap.read()
            if n % 15 == 0:
                frame_emotions = detector.top_emotion(frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = hog_face_detector(gray, 1)

                for face in faces:
                    shape = dlib_facelandmark(gray, face)
                    right_eye = shape_to_np_right(shape)
                    left_eye = shape_to_np_left(shape)

                    for (x, y) in right_eye[36:42]:
                        pos_eye_left.append([x,y])

                    for (x, y) in left_eye[42:48]:
                        pos_eye_right.append([x,y])
                        
                    emotions.append(frame_emotions)
            n += 1
        except:
            i += 1
            logging.info("Video processing ended!")

    cap.release()
    cv2.destroyAllWindows()

    std_left_eye = np.std(pos_eye_left)
    std_right_eye = np.std(pos_eye_right)
    format_std_left_eye = "{:.2f}".format(std_left_eye)
    format_std_right_eye = "{:.2f}".format(std_right_eye)

    results.append("Right eye's std: " +str(format_std_right_eye))
    results.append("Left eye's std: " + str(format_std_left_eye))
    results.append("Dominant emotions: " + str(emotions))
    
    data = results
    print(data)
    with open(f'processedFiles/{video_name}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    logging.info(f'Json file saved at: processedFiles/{video_name}.json')
