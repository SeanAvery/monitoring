import cv2

class VideoAnalyzer():
    def __init__(self, face):
        self.face = face

    ''' VIDEO PLAYER '''

    def process_video(self, video_file):
        video = cv2.VideoCapture('./videos/{}'.format(video_file))

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
