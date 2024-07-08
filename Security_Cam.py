#Team Code Disrupted
import cv2
import numpy as np
import winsound
#import os

# the next line is to convert .txt to no .txt as .txt can not be used. Uncomment and use the next line if coco.names is not getting detected
#os.rename(r"C:\Users\Tamabristi\Desktop\CODE\IT&PWskills_Hackathon\coco.names", "coco.names")


# Load the pre-trained YOLOv4-tiny model and class labels
net = cv2.dnn.readNet(r'C:\Users\Tamabristi\Desktop\Projects\IT&PWskills_Hackathon\yolov4-tiny.weights', r'C:\Users\Tamabristi\Desktop\Projects\IT&PWskills_Hackathon\yolov4-tiny.cfg.txt')
classes = []

#open coco.names in reading mode and call it f
with open(r"C:\Users\Tamabristi\Desktop\Projects\IT&PWskills_Hackathon\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Ask the user to select either an image or a video file
input_type = input("Enter 'image' or 'video' to select input type: ")

# If the input type is an image, read the image file
if input_type == "image":
    image_path = input("Enter the path to the image file: ")
    input_data = cv2.imread(image_path)

# If the input type is a video, read the video file or camera stream
elif input_type == "video":
    video_path = input("Enter the path to the video file, or enter '0' to use the camera: ")
    if video_path == "0":
        input_data = cv2.VideoCapture(0)
    else:
        input_data = cv2.VideoCapture(video_path)

# Initialize the face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Process the input data to detect faces
while True:
    # If the input type is an image, break the loop after processing once
    if input_type == "image":
        # Detect objects in the image using the YOLOv4-tiny model
        blob = cv2.dnn.blobFromImage(input_data, 1/255, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(net.getUnconnectedOutLayersNames())
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and (classes[class_id] == 'person' or classes[class_id] == 'bicycle' or classes[class_id] == 'car' or classes[class_id] == 'motorbike' or classes[class_id] == 'aeroplane' or classes[class_id] == 'bus' or classes[class_id] == 'train' or classes[class_id] == 'truck' or classes[class_id] == 'boat' or classes[class_id] == 'traffic light' or classes[class_id] == 'fire hydrant' or classes[class_id] == 'stop sign' or classes[class_id] == 'parking meter' or classes[class_id] == 'bench' or classes[class_id] == 'bird' or classes[class_id] == 'cat' or classes[class_id] == 'dog' or classes[class_id] == 'horse' or classes[class_id] == 'sheep' or classes[class_id] == 'cow' or classes[class_id] == 'elephant' or classes[class_id] == 'bear' or classes[class_id] == 'zebra' or classes[class_id] == 'giraffe' or classes[class_id] == 'backpack' or classes[class_id] == 'umbrella' or classes[class_id] == 'handbag' or classes[class_id] == 'tie' or classes[class_id] == 'suitcase' or classes[class_id] == 'frisbee' or classes[class_id] == 'skis' or classes[class_id] == 'snowboard' or classes[class_id] == 'sports ball' or classes[class_id] == 'kite' or classes[class_id] == 'baseball bat' or classes[class_id] == 'baseball glove' or classes[class_id] == 'skateboard' or classes[class_id] == 'surfboard' or classes[class_id] == 'tennis racket' or classes[class_id] == 'bottle' or classes[class_id] == 'wine glass' or classes[class_id] == 'cup' or classes[class_id] == 'fork' or classes[class_id] == 'knife' or classes[class_id] == 'spoon' or classes[class_id] == 'bowl' or classes[class_id] == 'banana' or classes[class_id] == 'apple' or classes[class_id] == 'sandwich' or classes[class_id] == 'orange' or classes[class_id] == 'broccoli' or classes[class_id] == 'carrot' or classes[class_id] == 'hot dog' or classes[class_id] == 'pizza' or classes[class_id] == 'donut' or classes[class_id] == 'cake' or classes[class_id] == 'chair' or classes[class_id] == 'sofa' or classes[class_id] == 'pottedplant' or classes[class_id] == 'bed' or classes[class_id] == 'diningtable' or classes[class_id] == 'toilet' or classes[class_id] == 'tvmonitor' or classes[class_id] == 'laptop' or classes[class_id] == 'mouse' or classes[class_id] == 'remote' or classes[class_id] == 'keyboard' or classes[class_id] == 'cell phone' or classes[class_id] == 'microwave' or classes[class_id] == 'oven' or classes[class_id] == 'toaster' or classes[class_id] == 'sink' or classes[class_id] == 'refrigerator' or classes[class_id] == 'book' or classes[class_id] == 'clock' or classes[class_id] == 'curtain' or classes[class_id] == 'photo frame' or classes[class_id] == 'vase' or classes[class_id] == 'scissors' or classes[class_id] == 'teddy bear' or classes[class_id] == 'hair drier' or classes[class_id] == 'toothbrush'):
                    center_x = int(detection[0] * input_data.shape[1])
                    center_y = int(detection[1] * input_data.shape[0])
                    width = int(detection[2] * input_data.shape[1])
                    height = int(detection[3] * input_data.shape[0])
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, width, height])
        # Detect faces in the cropped image and draw green rectangles
        for box in boxes:
            x, y, w, h = box
            face_img = input_data[y:y+h, x:x+w]
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            for (fx, fy, fw, fh) in faces:
                # Draw a green rectangle around each detected face
                cv2.rectangle(input_data, (x+fx, y+fy), (x+fx+fw, y+fy+fh), (0, 255, 0), 2)
        # Display the output image
        cv2.imshow("Output", input_data)
        cv2.waitKey(0)
        break
    
    # If the input type is a video, read the video file or camera stream
    elif input_type == "video":
        print("Note: After Entering the input again you will be redirected to a prompt where the detection will happen. To close the prompt press 'q'")
        print("Motion will be trakced using Green Box, Faces will de detected using Blue Box, Eyes will be detected using Red Box and Objects will be detected using Pink Box")
        print("Please ensure ample lighting for better results")
        video_path = input("Enter again(the path name) to comfirm: ")
        if video_path == "0":
            input_data = cv2.VideoCapture(0)
        else:
            input_data = cv2.VideoCapture(video_path)
        while True:
            # Read a frame from the video stream
            ret, frame1 = input_data.read()
            # If the frame could not be read, break the loop
            if not ret:
                break
            # Detect objects in the frame using the YOLOv4-tiny model
            blob = cv2.dnn.blobFromImage(frame1, 1/255, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            outs = net.forward(net.getUnconnectedOutLayersNames())
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    #check if the detected object is a person
                    if confidence > 0.5 and (classes[class_id] == 'person'):
                        #getting 2 frames of the same to compare 
                        ret, frame1 = input_data.read()
                        ret, frame2 = input_data.read()
                        #Loading the cascadeclassifier for the dataset to check if face or eye is there in the frame
                        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                        #calculating the absolute difference between the 2 frames
                        diff = cv2.absdiff(frame1, frame2)
                        #performing the check after transferring it to grayscale
                        gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
                        blur = cv2.GaussianBlur(gray, (5, 5), 0)
                        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
                        dilated = cv2.dilate(thresh, None, iterations=3)
                        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        #the next comment is to make contour lines
                        #cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)
                        for c in contours:
                            #decreasing the value in the next line will change how little of movement has to be captured
                            if cv2.contourArea(c) < 5000:
                                continue
                            #creating the rectangle around the person
                            x, y, w, h = cv2.boundingRect(c)
                            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 5)
                            #making sound for every movement
                            winsound.Beep(500, 200)
                        #detecting face by comparing with face cascade classifier
                        gray_1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray_1, 1.3, 5)
                        for (x, y, w, h) in faces:
                            cv2.rectangle(frame1, (x, y), (x + w, y + h), (255, 0, 0), 5)
                            roi_gray = gray_1[y:y+w, x:x+w]
                            roi_color = frame1[y:y+h, x:x+w]
                            #detecting face by comparing with eye cascade classifier
                            eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
                            for (ex, ey, ew, eh) in eyes:
                                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 5)              
      
                    #checking if any object is available
                    elif confidence > 0.5 and (classes[class_id] == 'bicycle' or classes[class_id] == 'car' or classes[class_id] == 'motorbike' or classes[class_id] == 'aeroplane' or classes[class_id] == 'bus' or classes[class_id] == 'train' or classes[class_id] == 'truck' or classes[class_id] == 'boat' or classes[class_id] == 'traffic light' or classes[class_id] == 'fire hydrant' or classes[class_id] == 'stop sign' or classes[class_id] == 'parking meter' or classes[class_id] == 'bench' or classes[class_id] == 'bird' or classes[class_id] == 'cat' or classes[class_id] == 'dog' or classes[class_id] == 'horse' or classes[class_id] == 'sheep' or classes[class_id] == 'cow' or classes[class_id] == 'elephant' or classes[class_id] == 'bear' or classes[class_id] == 'zebra' or classes[class_id] == 'giraffe' or classes[class_id] == 'backpack' or classes[class_id] == 'umbrella' or classes[class_id] == 'handbag' or classes[class_id] == 'tie' or classes[class_id] == 'suitcase' or classes[class_id] == 'frisbee' or classes[class_id] == 'skis' or classes[class_id] == 'snowboard' or classes[class_id] == 'sports ball' or classes[class_id] == 'kite' or classes[class_id] == 'baseball bat' or classes[class_id] == 'baseball glove' or classes[class_id] == 'skateboard' or classes[class_id] == 'surfboard' or classes[class_id] == 'tennis racket' or classes[class_id] == 'bottle' or classes[class_id] == 'wine glass' or classes[class_id] == 'cup' or classes[class_id] == 'fork' or classes[class_id] == 'knife' or classes[class_id] == 'spoon' or classes[class_id] == 'bowl' or classes[class_id] == 'banana' or classes[class_id] == 'apple' or classes[class_id] == 'sandwich' or classes[class_id] == 'orange' or classes[class_id] == 'broccoli' or classes[class_id] == 'carrot' or classes[class_id] == 'hot dog' or classes[class_id] == 'pizza' or classes[class_id] == 'donut' or classes[class_id] == 'cake' or classes[class_id] == 'chair' or classes[class_id] == 'sofa' or classes[class_id] == 'pottedplant' or classes[class_id] == 'bed' or classes[class_id] == 'diningtable' or classes[class_id] == 'toilet' or classes[class_id] == 'tvmonitor' or classes[class_id] == 'laptop' or classes[class_id] == 'mouse' or classes[class_id] == 'remote' or classes[class_id] == 'keyboard' or classes[class_id] == 'cell phone' or classes[class_id] == 'microwave' or classes[class_id] == 'oven' or classes[class_id] == 'toaster' or classes[class_id] == 'sink' or classes[class_id] == 'refrigerator' or classes[class_id] == 'book' or classes[class_id] == 'clock' or classes[class_id] == 'curtain' or classes[class_id] == 'photo frame' or classes[class_id] == 'vase' or classes[class_id] == 'scissors' or classes[class_id] == 'teddy bear' or classes[class_id] == 'hair drier' or classes[class_id] == 'toothbrush'):
                        center_x = int(detection[0] * frame1.shape[1])
                        center_y = int(detection[1] * frame1.shape[0])
                        width = int(detection[2] * frame1.shape[1])
                        height = int(detection[3] * frame1.shape[0])
                        x = int(center_x - width / 2)
                        y = int(center_y - height / 2)
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, width, height])
                        
                        # Draw a pink rectangle around each detected object
                        for box in boxes:
                            x, y, w, h = box
                            cv2.rectangle(frame1, (x, y), (x+w, y+h), (199, 5, 247), 4)

            #checking if q is pressed then exit the code            
            if cv2.waitKey(1) == ord('q'):
                break
            #showing the camera console to view the detection
            cv2.imshow('Camera', frame1)
        break

    # Release the video capture object and destroy all windows
    input_data.release()
    cv2.destroyAllWindows()