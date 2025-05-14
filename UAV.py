from ultralytics import YOLO
import cv2
import math 
from dronekit import connect, VehicleMode
# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
       #     x1, y1, x2, y2 = box.xyxy[0]
       #    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break
    #prioritize targets based on confidence score or proximity
    targets = []
    for box in boxes:
        x1,y1,x2,y2 = box.xyxy[0]
        confidence = math.ceil((box.conf[0]*100))/100
        cls = int (box.cls[0])
        target_data = {
            "class":classNames[cls],
            "confidence": confidence ,
            "x1":x1,
            "y1":y1,
            "x2":x2,
            "y2":y2
        }
        targets.append(target_data)

        #sort targets based on confidence 
        targets = sorted(targets, key=lambda x: x['confidence'], reverse=True)  # Fixed the key name
        if targets: 
            print("Prioritized Targets:", targets[0])  # Fixed indentation
        
        #initialize tracker
        tracker = cv2.TrackerKCF_create()  # Corrected the function name
        initBB = None

        #start Tracking the highest priority target
        if len(targets) > 0:
            top_target = targets[0]
            initBB = top_target["x1"], top_target["y1"], top_target["x2"], top_target["y2"]
            tracker.init(img, initBB)

        # target tracking in the loop
        if initBB is not None:
            success, box = tracker.update(img)
            if success:
                (x,y,w,h) = [int(v) for v in box]
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

cap.release()
cv2.destroyAllWindows()        
# INtegrating UAV control
#connect to the vehivle/UAV

vehicle = connect("udp:127.0.0.1:14550", wait_ready=True)  # Specify the correct UDP address

#Function to move UAV towards target

def move_towards_target(target_coordinates):
    #get current uav coordinates
    current_location = vehicle.location.global_relative_frame

    #calculate distance to the target 
    target_coordinates = (target_coordinates[0], target_coordinates[1], current_location.alt)
    distance = math.sqrt((target_coordinates[0] - current_location.lat)**2 + (target_coordinates[1]-current_location.lon)**2)