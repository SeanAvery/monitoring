import cv2

class FaceAnalyzer():
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('./Models/face_detector.xml')
        self.eye_cascade = cv2.CascadeClassifier('./Models/eye_detector.xml')

    def process_frame(self, frame):
        frame = self.detect_face(frame)
        # frame = self.detect_eyes(frame)
        return frame

    def detect_face(self, frame):
        faces = self.face_cascade.detectMultiScale(frame, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0, 95, 235), # uses BGR color format
                3)

        return frame
