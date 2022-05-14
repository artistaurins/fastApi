import cv2
import dlib
import json
import numpy as np
from fer import FER
from fer import Video

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

def is_user_concentrating(eye_pos, std):
    for i in range(len(eye_pos)):
        for j in range(i + 1, len(eye_pos)):
            if np.array(eye_pos[i]) - np.array(eye_pos[j]) > std:
                results.append("User is not paying attention")
                exit()
            else:
                results.append("User is paying attention")
                exit()

pos_eye_left = []
pos_eye_right = []
results = []
std_pos_left_eye = 0
std_pos_left_eye = 0
n = 1

def video_processing(video_name):
    cap = cv2.VideoCapture(video_name)
    video = Video(video_name)

    detector = FER(mtcnn=True)
    hog_face_detector = dlib.get_frontal_face_detector()
    dlib_facelandmark = dlib.shape_predictor("face_shape.dat")

    while True:
        _, frame = cap.read()
        if n % 10 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = hog_face_detector(gray, 1)
            
            for face in faces:
                face_landmarks = dlib_facelandmark(gray, face)
                shape = dlib_facelandmark(gray, face)
                right_eye = shape_to_np_right(shape)
                left_eye = shape_to_np_left(shape)

                for (x, y) in right_eye:
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), 1)
                for (x, y) in left_eye:
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), 1)
                
                for (x, y) in left_eye[36:42]:
                    pos_eye_left.append([x,y])

                for (x, y) in right_eye[42:48]:
                    pos_eye_right.append([x,y])

            key = cv2.waitKey(1)
            if key == 27:
                break

        n += 1

    cap.release()
    cv2.destroyAllWindows()

    std_pos_left_eye = np.std(pos_eye_left)
    std_pos_right_eye = np.std(pos_eye_right)

    is_user_concentrating(pos_eye_left, std_pos_left_eye)
    is_user_concentrating(pos_eye_right, std_pos_right_eye)

    raw_data = video.analyze(detector, display=True, frequency=160, save_video=False, save_frames=False)
    df = video.to_pandas(raw_data)

    results.append(df)

    data = results
    with open('processedFiles/{video_filename}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
