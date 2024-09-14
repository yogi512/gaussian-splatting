from argparse import ArgumentParser
import sys
import os
import cv2
from PIL import Image

def extract_frames(video_path, start_count, interval, output_dir):
    video_frames = cv2.VideoCapture(video_path)
    count = start_count
    frame_count = 0
    while video_frames.isOpened():
        ret, video_frame = video_frames.read()
        if not ret:
            break

        if frame_count % interval == 0:
            video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
            video_frame = Image.fromarray(video_frame)
            video_frame.save(os.path.join(output_dir, "frame_%05d.png" % count))
            count += 1

        frame_count += 1

    video_frames.release()
    return count

if __name__ == '__main__':
    parser = ArgumentParser(description="Extract images from dynerf videos")
    parser.add_argument("--datadir", default='data/custom/work', type=str)
    args = parser.parse_args()
    print(args.datadir)

    sc_name = str(args.datadir).split('/')[-1]

    # Create a single directory to store all images
    output_dir = os.path.join(args.datadir, 'input')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process work_exo video first
    try:
        exo_video_path = os.path.join(args.datadir, sc_name+'_exo.mov')
    except: 
        exo_video_path = os.path.join(args.datadir, sc_name+'_exo.mp4')
    exo_video_frames = cv2.VideoCapture(exo_video_path)
    fps = exo_video_frames.get(cv2.CAP_PROP_FPS)
    interval = int(fps / 5)  # Calculate the interval to get 5 frames per second

    start_count = 1
    next_count = extract_frames(exo_video_path, start_count, interval, output_dir)
    print("Frames from work_exo extracted and saved successfully.")
    print("Next count: ", next_count)

    try:
    # Process work_ego video next
        try:
            ego_video_path = os.path.join(args.datadir, sc_name+'_ego.mov')
        except:
            ego_video_path = os.path.join(args.datadir, sc_name+'_ego.mp4')
            
        extract_frames(ego_video_path, next_count, interval, output_dir)
        print("Frames from work_ego extracted and saved successfully.")
    except: 
        print("No work_ego video found.")