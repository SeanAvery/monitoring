import cv2

class VideoPlayer():
    def __init__(self, model_path, face_model, eye_model):
        self.face_cascade = cv2.CascadeClassifier('./Models/face_detector.xml')
        self.eye_cascade = cv2.CascadeClassifier('./Models/eye_detector.xml')

    def detect_faces(self, frame):
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(frame, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        return frame

    def detect_eyes(self, frame):
        eyes = self.eye_cascade.detectMultiScale(frame)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        return frame

    def process_frame(self, frame):
        return self.detect_eyes(self.detect_faces(frame))

    def process_video(self, video_file):
            video = cv2.VideoCapture('./videos/{}'.format(video_file))

            while(True):

                ret, frame = video.read()

                if ret == True:
                    procd_image = self.process_frame(frame)
                    cv2.imshow('frame', procd_image)
                    cv2.waitKey(1)

                else:
                    break

            video.release()
            cv2.destroyAllWindows()
