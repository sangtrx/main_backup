import cv2
import numpy as np
from collections import Counter

# Read the video file
cap = cv2.VideoCapture('/mnt/tqsang/vid/WIN_20230217_09_25_25_Pro.mp4')

# Initialize an empty list to store histograms
histograms = []

# Loop through each frame in the video
while cap.isOpened():
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate the histogram for the grayscale image
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    
    # Append the histogram to the list of histograms
    histograms.append(hist)

# Find the most common histogram
most_common_hist = Counter(map(tuple, histograms)).most_common(1)[0][0]

# Convert the most common histogram to a numpy array
most_common_hist = np.array(most_common_hist)

# Normalize the histogram to the range [0, 255]
most_common_hist = cv2.normalize(most_common_hist, None, 0, 255, cv2.NORM_MINMAX)

# Convert the histogram to a 3-channel RGB image
reference_frame = np.zeros((100, 256, 3), np.uint8)
reference_frame[:,:,0] = most_common_hist
reference_frame[:,:,1] = most_common_hist
reference_frame[:,:,2] = most_common_hist

# Initialize a VideoWriter object to write the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/mnt/tqsang/vid/output_video.mp4', fourcc, 30, (reference_frame.shape[1], reference_frame.shape[0]))

# Loop through each frame in the video again and match its histogram to the reference histogram
cap = cv2.VideoCapture('/mnt/tqsang/vid/WIN_20230217_09_25_25_Pro.mp4')
while cap.isOpened():
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Match the histogram of the frame to the reference histogram
    frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    frame_yuv[:,:,0] = cv2.equalizeHist(frame_yuv[:,:,0])
    frame = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2BGR)
    
    # Write the frame to the output video
    out.write(frame)

# Release the VideoCapture and VideoWriter objects
cap.release()
out.release()
