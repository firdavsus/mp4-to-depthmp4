# processVideo.py

Everything was created by ChatGPT or is from MiDaS

This Python script processes video files to generate depth map videos using the MiDaS model. The script supports input videos in .mp4, .mkv, .mov, and .avi formats, and the output depth map videos will be in .mp4 format.

# Setup

I have only tested this on windows with a 3090, it requires atleast 20GB of vram, but that can be turned down by changing the batch_size in run.py

1. install anaconda
2. clone this repo
3. download the weights to ./weights/ from [here](https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt).
4. open the anaconda terminal as administrator
5. run setup.bat to setup the conda env

# Usage

Create the following directories in the same location as the script:
* ./unformatted_videos/: Place your input videos in this folder.
* ./depth_videos/: The script will store the generated depth map videos in this folder.

run processVideo.py to process all the videos. It takes around 2x the duration of the video depending on the resolution