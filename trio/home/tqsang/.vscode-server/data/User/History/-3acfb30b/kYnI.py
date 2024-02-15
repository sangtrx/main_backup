import cv2
import numpy as np

# Read the input video
cap = cv2.VideoCapture('/mnt/tqsang/vid/WIN_20230217_09_25_25_Pro.mp4')

# Read the input image
ref_img = cv2.imread('/mnt/tqsang/vid/frame_mau.png')

# Convert the input image to grayscale
ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

# Calculate the reference histogram
ref_hist, _ = np.histogram(ref_img, 256, [0, 256])

# Define the output video file
out = cv2.VideoWriter('/mnt/tqsang/vid/output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (int(cap.get(3)), int(cap.get(4))))

# Loop through all the frames in the input video
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        # Convert the current frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate the current frame histogram
        curr_hist, _ = np.histogram(gray_frame, 256, [0, 256])

        # Perform histogram matching
        lut = np.interp(np.linspace(0, 255, 256), np.linspace(0, 255, 256), np.cumsum(ref_hist) * 255 / ref_img.size).astype('uint8')
        matched_frame = cv2.LUT(gray_frame, lut)

        # Convert the matched frame back to BGR format
        matched_frame = cv2.cvtColor(matched_frame, cv2.COLOR_GRAY2BGR)

        # Write the output frame to the video file
        out.write(matched_frame)

        # # Display the current frame
        # cv2.imshow('frame', matched_frame)

        # # Press 'q' to exit
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    else:
        break

# Release the resources
cap.release()
out.release()
cv2.destroyAllWindows()
