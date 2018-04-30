import cv2
import dlib
import numpy as np

class Face():
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('./Models/dlib_facial_lm.dat')
        self.init_3d_model()

    def init_3d_model(self):
        self.model_points = np.array([
            (0.0, 0.0, 0.0),            # tip of nose
            (0.0, -330.0, -65.0),       # tip of chin
            (-225.0, 170.0, -135.0),    # left, left eye corner
            (225.0, 170.0, -135.0),     # right, right eye corner
            (-150.0, -150, -125.0),     # left mouth corner
            (150.0, -150.0, 125.0)      # right mouth corner
        ])

    def set_camera_matrix(self, camera_matrix):
        self.camera_matrix = camera_matrix
        # distance coefficients
        self.dist_coeffs = np.zeros((4, 1))

    ''' FACE ANALYZER '''

    def analyze_face(self, frame):
        faces = self.detector(frame, 1)
        face = self.reduce_faces(faces)
        features = self.detect_features(frame, face)
        reduced_features = self.reduce_features(features)
        rotation, translation = self.calculate_pose(reduced_features)
        self.draw_head_pose_vector(frame, rotation, translation, reduced_features)
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
        features = self.shape_to_np(raw_shape)

        # draw landmarks on frame
        for (x, y) in features:
            cv2.circle (frame, (x, y), 2, (0, 95, 235), -1)

        return features

    def calculate_pose(self, image_points):
        (success, rotation, translation) = cv2.solvePnP(
            self.model_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE)
        if success == True:
            return (rotation, translation)
        else:
            raise ValueError('could not calculate rotation, translation')

    def reduce_features(self, features):
        return np.array([
            tuple(features[30]),     # nose tip
            tuple(features[8]),      # chin tip
            tuple(features[36]),     # left corner, left eye
            tuple(features[46]),     # right corner, right eye
            tuple(features[48]),     # left corner, mouth
            tuple(features[54])      # right corner, mouth
        ], dtype='double')

    def draw_head_pose_vector(self, frame, rotation, translation, features):
        (head_pose, jacobian) = cv2.projectPoints(
            np.array([(0.0, 0.0, 1000.0)]),
            rotation,
            translation,
            self.camera_matrix,
            self.dist_coeffs)

        p1 = ( int(features[0][0]), int(features[0][1]))
        p2 = ( int(head_pose[0][0][0]), int(head_pose[0][0][1]))

        cv2.line(frame, p1, p2, (255, 0, 0), 2)

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
