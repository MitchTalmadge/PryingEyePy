import os
import time

import face_recognition

from src.face import FaceRepository, Face
import cv2
import easygui

def add_face():
    video_capture = cv2.VideoCapture(0)

    milliseconds_to_wait = 60
    while milliseconds_to_wait > 0:
        ret, frame = video_capture.read()

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, f"{milliseconds_to_wait / 100 * 2}", (5, 30), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.001)
        milliseconds_to_wait -= 1

    ret, frame = video_capture.read()
    name = easygui.enterbox("Name:")

    cv2.imwrite("tmp.png", frame)

    video_capture.release()
    cv2.destroyAllWindows()

    face_repo = FaceRepository()

    face_img = face_recognition.load_image_file("tmp.png")
    face_data = face_recognition.face_encodings(face_img)[0]

    face_repo.save(
        Face(name, face_data)
    )

    os.remove("tmp.png")


if __name__ == '__main__':
    add_face()