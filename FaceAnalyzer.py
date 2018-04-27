import cv2

class FaceAnalyzer():
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('./Models/face_detector.xml')
        self.eye_cascade = cv2.CascadeClassifier('./Models/eye_detector.xml')

    def process_frame(self, frame):
        frame = self.detect_face(frame)
        return frame

    def detect_face(self, frame):
        faces = self.face_cascade.detectMultiScale(frame, 1.3, 5)

        # draw rectangle on frame
        for (x, y, w, h) in faces:
            self.closeup_coords = (x, y, w, h)
            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0, 95, 235), # uses BGR color format
                3)

            closeup = frame[y: y+h, x: x+w]

            self.detect_eyes(frame, closeup, (x, y, w, h))

        # crop out face closeup
        return frame

    def detect_eyes(self, frame, closeup, coords):
        eyes = self.eye_cascade.detectMultiScale(closeup)

        for (ex, ey, ew, eh) in eyes:
            # draw rectangle on large frame
            cv2.rectangle(
                frame,
                (coords[0] + ex, coords[1] + ey),
                (coords[0] + ex + ew, coords[1] + ey + eh),
                (20, 255, 57),
                3)

        cv2.imshow('frame', frame)

        return frame
