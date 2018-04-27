from VideoPlayer import VideoPlayer

model_path = '/Models'
face_model = 'face_detector.xml'
eye_model = 'eye_detector.xml'
video_file = 'video_2.hevc'

if __name__ == '__main__':
    player = VideoPlayer(model_path, face_model, eye_model)
    player.process_video(video_file)
