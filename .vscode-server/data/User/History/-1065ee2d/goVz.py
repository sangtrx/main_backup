import cv2
import numpy as np

# Read the input video
cap = cv2.VideoCapture('/mnt/tqsang/vid/WIN_20230217_09_25_25_Pro.mp4')

# Initialize dictionary to store histogram counts
hist_counts = {}

# Loop through all frames
while cap.isOpened():
    # Read a frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate the histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0,256])
    
    # Convert the histogram to a tuple to use as a dictionary key
    hist_key = tuple(hist.reshape(-1))
    
    # Add the histogram to the dictionary or update its count
    hist_counts[hist_key] = hist_counts.get(hist_key, 0) + 1

# Find the most frequent histogram
most_frequent_hist = max(hist_counts, key=hist_counts.get)

# Convert the tuple histogram back to a 256-bin array
most_frequent_hist = np.array(most_frequent_hist).reshape((256,))

# Reset the video capture to the beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Initialize the output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/mnt/tqsang/vid/output_video.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# Loop through all frames again and apply histogram matching
while cap.isOpened():
    # Read a frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Match the frame histogram to the most frequent histogram
    matched = cv2.LUT(frame, cv2.calcHist([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0,256]))

    # Write the matched frame to the output video
    out.write(matched)

# Release the video capture and writer and close the output window
cap.release()
out.release()
cv2.destroyAllWindows()
