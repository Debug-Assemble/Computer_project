import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
model = load_model(r'C:\Users\soumi\Downloads\hand-gesture-recognition-code\mp_hand_gesture')
# Load class names
labels = ['okay', 'peace', 'thumbs up', 'thumbs down', 'call me', 'stop', 'rock', 'live long', 'fist', 'smile']
print(labels)
cap = cv2.VideoCapture(0)
while True:
    # Read each frame from the webcam
    _, frame = cap.read()
    x, y, c = frame.shape
    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Get hand landmark prediction
    result = hands.process(framergb)
    # Show the final output
    class_name = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])
            # Drawing landmarks on frames
            mp_draw.draw_landmarks(frame, handslms, mp_hands.HAND_CONNECTIONS)
            # Predict gesture in Hand Gesture Recognition project
            prediction = model.predict([landmarks])
            # print(prediction)
            classID = np.argmax(prediction)
            class_name = labels[classID].capitalize()
            # show the prediction on the frame
        cv2.putText(frame, class_name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Output", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    # release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
