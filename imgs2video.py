import os
import cv2
import argparse

def get_image_files(datadir):
    # Get list of image files in the directory
    image_files = [os.path.join(datadir, f) for f in os.listdir(datadir) if f.endswith(('png', 'jpg', 'jpeg'))]
    image_files.sort()  # Sort to maintain the order
    return image_files

def create_video_from_images(datadir, output_file='cs1.mp4', fps=30):
    image_files = get_image_files(datadir)
    
    if not image_files:
        print("No images found in the directory.")
        return

    # Read the first image to get the dimensions
    frame = cv2.imread(image_files[0])
    height, width, _ = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for image_file in image_files:
        frame = cv2.imread(image_file)
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved as {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert images in a directory to a MP4 video.")
    parser.add_argument('--datadir', type=str, default='output/work_1/test/ours_30000/renders',required=True, help="Path to the directory containing images.")
    parser.add_argument('--output', type=str, default='output.mp4', help="Output video file name.")
    parser.add_argument('--fps', type=int, default=30, help="Frames per second for the video.")

    args = parser.parse_args()
    
    create_video_from_images(args.datadir, args.output, args.fps)