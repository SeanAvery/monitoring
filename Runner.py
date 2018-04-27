import os
import cv2 

def watch_video(name):
    video = cv2.VideoCapture('./videos/{}'.format(name))
    
    while(True):
        
        ret, frame = video.read()
        
        if ret == True:
            cv2.imshow('frame', frame)
            cv2.waitKey(1)

        else: 
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # read video
    watch_video('video_4.hevc')
