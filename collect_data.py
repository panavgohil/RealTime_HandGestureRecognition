import cv2
import csv
import os
import mediapipe as mp
cap = cv2.VideoCapture(0) # connecting to webcam, 0 means default webcam
mp_hands = mp.solutions.hands # hand module
hands = mp_hands.Hands() # hand detector object
csv_file = "dataset.csv"

if not os.path.exists(csv_file): # create dataset.csv if it doesn't exist
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        header = []        
        for i in range(21):
            header.append(f"x{i}")
            header.append(f"y{i}")        
           
        header.append("label") # gesture label column
        writer.writerow(header)
while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    # convert BGR to RGB for mediapipe
    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )    
    results = hands.process(frame_rgb)    
    landmarks = []
    if results.multi_hand_landmarks:
        # take only one detected hand
        hand = results.multi_hand_landmarks[0]  
        base_x=hand.landmark[0].x
        base_y=hand.landmark[0].y
           
        for lm in hand.landmark:
            landmarks.append(lm.x-base_x)
            landmarks.append(lm.y-base_y)   
             
        print("Number of features:", len(landmarks))
        

   
    cv2.imshow( #webcam feed
        "Webcam",
        frame
    )
    

    
    key = cv2.waitKey(1) &0xFF
    if key==ord('o') and len(landmarks)==42:
            landmarks.append('open')
            with open(csv_file,'a',newline="") as f:
                writer=csv.writer(f)
                writer.writerow(landmarks)
            print("Saved, open palm")
    elif key==ord('f') and len(landmarks)==42:
            landmarks.append('fist')
            with open(csv_file,'a',newline="") as f:
                writer=csv.writer(f)
                writer.writerow(landmarks)
            print("Saved, fist")
    elif key==ord('p') and len(landmarks)==42:
            landmarks.append('point')
            with open(csv_file,'a',newline="") as f:
                writer=csv.writer(f)
                writer.writerow(landmarks)
            print("Saved, point")
    elif key==ord('t') and len(landmarks)==42:
            landmarks.append('thumbsUp')
            with open(csv_file,'a',newline="") as f:
                writer=csv.writer(f)
                writer.writerow(landmarks)
            print("Saved, ThumbsUp")
    if key==27:
        
        break
cap.release()
cv2.destroyAllWindows()