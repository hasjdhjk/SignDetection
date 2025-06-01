# 
#
# This file is a checker for individual files of recordings in landmark_data. 
#
#


import numpy as np
data = np.load('landmark_data/(label name)/landmark_(label name)_1.npy')

print(data[0:5])
print(len(data))                               # Should be fps * video duration. 
print(len(data[0]))                            # Should be 129 (64 values of each hand (21 landmarks, 3D, 1 confidence), and 1 timestamp). 