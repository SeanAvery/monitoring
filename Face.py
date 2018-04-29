import cv2
import dlib
import numpy as np

class Face():
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('./Models/dlib_facial_lm.dat')

    ''' FACE ANALYZER '''

    def analyze_face(self, frame):
        faces = self.detector(frame, 1)
        face = self.reduce_faces(faces)
        self.detect_features(frame, face)
        return frame

    def reduce_faces(self, faces):
        if len(faces) == 0:
            return None
        if len(faces) == 1:
            return faces[0]
        if len(faces) > 1:
            return faces[0]

    def detect_features(self, frame, face):
        # get facial landmarkd from dlib model
        raw_shape = self.predictor(frame, face)

        # convert features to a coordinate array
        shape = self.shape_to_np(raw_shape)

        # draw landmarks on frame
        for (x, y) in shape:
            cv2.circle (frame, (x, y), 2, (0, 95, 235), -1)


    ''' UTILS '''

    def shape_to_np(self, shape):
        coords = np.zeros((68, 2), dtype='int')

        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        return coords

    def convert_to_rect(self, frame):
        x = frame.left()
        y = frame.top()
        w = frame.right() - x
        h = frame.bottom() - y
