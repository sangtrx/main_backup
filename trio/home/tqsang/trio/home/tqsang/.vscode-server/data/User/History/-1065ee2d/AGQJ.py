import cv2
import numpy as np

# Load video
cap = cv2.VideoCapture('/mnt/tqsang/vid/WIN_20230217_09_25_25_Pro.mp4')

# Calculate histogram for each frame
histograms = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_frame], [0, 1], None, [180, 256], [0, 180, 0, 256])
    histograms.append(hist)

# Select major frame histogram as reference histogram
flat_histograms = [hist.flatten() for hist in histograms]
histogram_frequencies = np.bincount(np.argmax(flat_histograms, axis=0))
reference_histogram = histograms[np.argmax(histogram_frequencies)]

# Normalize histograms of all frames using reference histogram
for i, hist in enumerate(histograms):
    histograms[i] = cv2.normalize(hist, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    histograms[i] = histograms[i].flatten()

# Write processed frames to new video file
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/mnt/tqsang/vid/output_video.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
for i in range(len(histograms)):
    ret, frame = cap.read()
    if not ret:
        break
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    histograms[i] = cv2.normalize(histograms[i], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    histograms[i] = histograms[i].reshape(-1, 1, 1)
    hist_mask = cv2.calcBackProject([hsv_frame], [0, 1], reference_histogram, [0, 180, 0, 256], 1)
    hist_mask = cv2.merge([hist_mask, hist_mask, hist_mask])
    filtered_frame = cv2.bitwise_and(frame, hist_mask)
    out.write(filtered_frame)

cap.release()
out.release()
