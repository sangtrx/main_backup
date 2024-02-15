import cv2

# Load the video file
video = cv2.VideoCapture("input_video.mp4")

# Select the major frame histogram as the reference
ref_hist = None
major_frame_index = None
major_frame_count = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the histogram of the grayscale frame
    hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])

    # Check if the current frame is the major frame
    if hist.sum() > major_frame_count:
        major_frame_index = video.get(cv2.CAP_PROP_POS_FRAMES)
        ref_hist = hist
        major_frame_count = hist.sum()

# Reset the video to the beginning
video.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Loop through all the frames and match their histograms to the reference
while True:
    ret, frame = video.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Match the histogram of the grayscale frame to the reference histogram
    matched_frame = cv2.LUT(gray_frame, cv2.normalize(ref_hist, None, 0, 255, cv2.NORM_MINMAX))

    # Convert the matched frame back to RGB
    output_frame = cv2.cvtColor(matched_frame, cv2.COLOR_GRAY2BGR)

# Release the video and close the windows
video.release()
cv2.destroyAllWindows()
