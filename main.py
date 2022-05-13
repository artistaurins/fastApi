import cv2
import dlib
import numpy as np
from fer import FER
from fer import Video
from matplotlib import pyplot as plt

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

def video_processing(video_name):
    video_filename =f"processedFiles/{video_name}.mp4"
    cap = cv2.VideoCapture(video_filename)
    # video = Video(video_filename)

    # detector = FER(mtcnn=True)
    hog_face_detector = dlib.get_frontal_face_detector()
    dlib_facelandmark = dlib.shape_predictor("face_shape.dat")

    pos_eye_left = []
    pos_eye_right = []
    n = 1

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
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), 1)
                for (x, y) in left_eye:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), 1)
                
                for (x, y) in right_eye[36:42]:
                    pos_eye_left.append([x,y])

                for (x, y) in left_eye[42:48]:
                    pos_eye_right.append([x,y])

            # cv2.imshow("Face Landmarks", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
        n += 1

    cap.release()
    cv2.destroyAllWindows()

    # data = video.analyze(detector, display=False, frequency=30, save_video=False, save_frames=False)
    # df = video.to_pandas(data)
    # df = video.get_emotions(df)
    # df.plot()
    # plt.show()
    # emotion, score = detector.top_emotion(df)

    std_pos_left_eye = np.std(pos_eye_left)

    std_pos_right_eye = np.std(pos_eye_right)

    # Acu sakuma pozicija
    starting_pos_left_eye = str(pos_eye_left[0])

    print("-------------------------")

    print("Kreisas acs sakuma postion: " + starting_pos_left_eye)

    print("Labas acs std X: " + str(std_pos_left_eye))
    print("Kreisas acs std: " + str(std_pos_right_eye))
