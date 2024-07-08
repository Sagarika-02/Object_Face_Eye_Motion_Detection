import cv2

# Try to open the default camera (index 0)
cap = cv2.VideoCapture(0)

# Check if the video capture object is opened successfully
if cap.isOpened():
    print("Camera is accessible.")
    # Release the video capture object
    cap.release()
else:
    print("Error: Unable to access the camera.")
