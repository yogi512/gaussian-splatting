from argparse import ArgumentParser
import sys 
import os 
import cv2
import os
from PIL import Image


if __name__ == '__main__':
    parser = ArgumentParser(description="Extract images from dynerf videos")
    parser.add_argument("--datadir", default='data/multipleview/iiith_cooking_54_4', type=str)
    args = parser.parse_args()
    print(args.datadir)
    for video in os.listdir(args.datadir):
        print(video)
        video_path = os.path.join(args.datadir, video)
        try:
            video_frames = cv2.VideoCapture(video_path)
        except:
            pass
        # Get the frame rate of the video
        fps = video_frames.get(cv2.CAP_PROP_FPS)
        interval = int(fps / 5)  # Calculate the interval to get 2 frames per second

        count = 1
        frame_count = 0
        video_images_path = video_path.split('.')[0]
        image_path = os.path.join(video_images_path)

        if not os.path.exists(image_path):
            os.makedirs(image_path)

        while video_frames.isOpened():
            ret, video_frame = video_frames.read()
            if not ret:
                break
            
            if frame_count % interval == 0:
                video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
                video_frame = Image.fromarray(video_frame)
                video_frame.save(os.path.join(image_path, "frame_%05d.png" % count))
                count += 1
            
            frame_count += 1

        video_frames.release()
        print("Frames extracted and saved successfully.")