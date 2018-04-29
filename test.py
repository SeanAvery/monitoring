from VideoAnalyzer import VideoAnalyzer
from Face import Face

video_file = 'video_2.hevc'

if __name__ == '__main__':
    face = Face()
    video = VideoAnalyzer(face)
    video.process_video(video_file)


  
