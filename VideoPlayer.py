import cv2

class VideoPlayer():
    def __init__(self, FaceAnalyzer):
        self.FaceAnalyzer = FaceAnalyzer

    def process_video(self, video_file):
        video = cv2.VideoCapture('./videos/{}'.format(video_file))

        while(True):

            ret, frame = video.read()

            if ret == True:
                procd_image = self.FaceAnalyzer.process_frame(frame)
                cv2.imshow('frame', procd_image)
                cv2.waitKey(1)

            else:
                break

        video.release()
        cv2.destroyAllWindows()
