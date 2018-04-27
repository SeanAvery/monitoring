from VideoPlayer import VideoPlayer
from FaceAnalyzer import FaceAnalyzer

video_file = 'video_2.hevc'

if __name__ == '__main__':
    analyzer = FaceAnalyzer()
    player = VideoPlayer(analyzer)
    player.process_video(video_file)
