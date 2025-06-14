# 
#
# This file is a checker for individual files of recordings in landmark_data. 
#
#


import numpy as np
data = np.load('landmark_data/hello/landmark_hello_1.npy')

print(data[0:5])
print(len(data))                               # Should be fps * video duration. 
print(len(data[0]))                            # Should be 129 (64 values of each hand (21 landmarks, 3D, 1 confidence), and 1 timestamp). 