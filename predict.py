import cv2
import mediapipe as mp
import pandas as pd
import joblib

model=joblib.load("gesture_model.pkl")

cap=cv2.VideoCapture(0)
mp_hands=mp.solutions.hands
hands=mp_hands.Hands()

while True:
    success, frame=cap.read()
    if not success:
        break
    frame=cv2.flip(frame,1)
    frame_rgb=cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB

    )
    results=hands.process(frame_rgb)
    landmarks=[]
    if results.multi_hand_landmarks:
        hand=results.multi_hand_landmarks[0]
        
        base_x = hand.landmark[0].x
        base_y = hand.landmark[0].y

        for lm in hand.landmark:
            landmarks.append(lm.x - base_x)
            landmarks.append(lm.y - base_y)
            
        print(len(landmarks))
        if len(landmarks)==42:
            input_data=pd.DataFrame([landmarks])
            prediction=model.predict(input_data)
            probabilities=model.predict_proba(input_data)
            gesture=prediction[0]
            confidence=max(probabilities[0])*100
            cv2.putText(
                frame,
                f"{gesture} {confidence:.1f}%",
                (50,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,0,0),
                2

            )
            



    cv2.imshow(
        "prediction",
        frame
    )
    if cv2.waitKey(1)==27:
        break
cap.release()
cv2.destroyAllWindows