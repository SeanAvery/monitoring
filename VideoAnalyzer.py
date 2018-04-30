import cv2
import numpy as np

class VideoAnalyzer():
    def __init__(self, face):
        self.face = face

    ''' VIDEO PLAYER '''

    def process_video(self, video_file):
        video = cv2.VideoCapture('./videos/{}'.format(video_file))

        # "callibrate" camera
        self.estimate_camera_calibration(video)

        # run video through processor and display output
        while(True):
            ret, frame = video.read()
            if ret == True:
                processed_image = self.process_frame(frame)
                cv2.imshow('output', processed_image)
                cv2.waitKey(1)

            else:
                break

        video.release()
        cv2.destroyAllWindows()

    ''' PROCESSOR '''

    def process_frame(self, frame):
        frame = self.face.analyze_face(frame)
        # frame, closeup = self.face.get_face(frame, portrait)
        # frame, left_eye, right_eye = self.get_eyes(frame, closeup)
        return frame

    ''' CALIBRATION '''

    def estimate_camera_calibration(self, video):
        ret, frame = video.read()

        if ret == True:
            frame_size = frame.shape
            focal_length = frame_size[1]
            center = (frame_size[1]/2, frame_size[0]/2)

            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                [0, focal_length,  center[1]],
                [0, 0, 1]],
                dtype="double")

            self.face.set_camera_matrix(camera_matrix)

        else:
            raise ValueError('could not read frame from video')
