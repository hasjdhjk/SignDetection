This is a working area for all the software for Hephaistos.

Sign Language dataset:
https://imperiallondon-my.sharepoint.com/:f:/r/personal/at2123_ic_ac_uk/Documents/Hephaistos?csf=1&web=1&e=c9XmRr 



#---------------------------------------------#
# INSTRUCTIONS                                #
#                                             #
#---------------------------------------------#
To reproduce: 

1. Run the file 'data_recorder.py' in the 'mediapipe' folder. This will access the camera on your device and overlay the MediaPipe landmarks live. 

2. Stay within the live feed and press 'r'. You will see a countdown on your screen. A recording, and its corresponding file of landmarks, will be made. The video will be saved in the 'recorded_videos' folder. The landmark data will be saved within the 'landmark_data' folder. 

3. Press 'q' to quit the screen. 

4. You can alter where your files are saved by changing the following variables in the 'DIRECTORY AND FILE SET-UP' in 'data_recorder.py': 
    video_dir = "recorded_videos/(label name)"
    landmark_dir = "landmark_data/(label name)"
    video_title = "video_(label name)_(recording number).mp4"
    landmark_title = "landmark_(label name)_(recording number).npy"

5. Once all videos are recorded, go to the 'training' folder and run 'train_LSTM.py'. This will take care of the data and train the LSTM. 

6. Have big funs! 




#---------------------------------------------#
# THINGS TO BE AWARE OF                       #
#                                             #
#---------------------------------------------#

1. The variables 'self.sample_interval' in hand_tracker.py and 'fps' in the main script of data_recorder.py SHOULD BE CONGRUENT. 