
#scp -r '/Users/artem/Desktop/Code/School Year 2/AAC' aac@aac.local:/home/aac

#scp -r  aac@aac.local:/home/aac/AAC/videos ~/Downloads

import os
import pandas as pd
import time

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from libcamera import Transform

from pyfiglet import Figlet

EXCEL_FILE = "labels.xlsx"
OUTPUT_DIR = "videos"
LABEL_COL_NAME = 'A'

NUM_REPEATS = 3

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Video settings
RESOLUTION = (1296, 972)  # Set resolution (width, height)
BITRATE = 10000000  # Set bitrate (10000000 for 10 Mbps)
FRAMERATE = 30
VERTICAL_FLIP = True

def load_labels(file_path):
    try:
        df = pd.read_excel(file_path)
        return df[LABEL_COL_NAME].tolist()
    except Exception as e:
        print(f"Error loading labels: {e}")
        return []

def main():

    os.system("clear")

    #ASCII gen
    f = Figlet(font='slant')

    # Initialize camera
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": RESOLUTION}, transform=Transform(vflip=VERTICAL_FLIP, hflip=True))
    picam2.configure(config)

    frame_duration = int(1e6 / FRAMERATE)  # Frame duration in microseconds
    picam2.set_controls({"FrameDurationLimits": (frame_duration, frame_duration), "Brightness": 0.15})

    encoder = H264Encoder(BITRATE)

    video_index = 32

    # Load the labels
    labels = load_labels(EXCEL_FILE)
    
    if not labels:
        print("No labels found. Ensure the Excel file is formatted correctly with a 'Label' column.")
        return

    print(f"Loaded {len(labels)} labels. Starting recording process...")

    i = 0

    while (video_index < len(labels)):

        label = str(labels[video_index])


        os.system("clear")


        print(f"{video_index} / {len(labels)} done.")
        print(f"{i} of {NUM_REPEATS} this sign recorded.")
        print("----------------------")

        print("RECORDING NOT STARTED.")
        print("----------------------")
        print()

        print(f.renderText(label))
        print()

        print("Press <ENTER> to BEGIN recording.")

        input1 = str(input())

        try:
            if input1 == "":

                i+=1

                video_filename = os.path.join(OUTPUT_DIR, f"{label}_{i}.h264")

                picam2.start_recording(encoder, video_filename)

                print(".\n")
                time.sleep(1) #seconds to allow compression to adjust 
                print(".\n")
                time.sleep(1)
                print(".\n")
                print(f"Recording started for label: {label}")
                print()
                print("-+-+-+-+-+-+-+-+-+-+-+-")
                print("RECORDING IN PROGRESS...")
                print("-+-+-+-+-+-+-+-+-+-+-+-")
                print()
                print("Press <ENTER> to STOP recording.")

                input2 = input()
                while input2 != "":
                    input2 = input()
                
                # Stop recording
                picam2.stop_recording()

                if i == NUM_REPEATS:
                    video_index += 1
                    i = 0

                print()
                print()
                print()
                print("RECORDING STOPPED SUCCESSFULLY!")
                print()
                print("Press <Enter> to move on.")

                input2 = input()
                while input2 != "":
                    input2 = input()
                    
        except Exception as e:
            print(f"Error during recording: {e}")

    # Cleanup
    picam2.close()
    print("Recording process completed.")

if __name__ == "__main__":
    main()
