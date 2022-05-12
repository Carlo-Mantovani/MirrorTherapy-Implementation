import cv2
from cv2 import sqrt
from cv2 import norm
import mediapipe as mp
import time
import math
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    x = 250
    y = 250
    center_coordinates = (x, y)
    radius = 15
    color = (0, 0, 255)
    thickness = -1
    ok = "ok"

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):

                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y*h)
                coordinates = (cx, cy)
                c1 = x-cx
                c2 = y-cy

                distancia = math.sqrt(c1**2 + c2**2)

                cv2.circle(img, coordinates, 5, (255, 0, 0), cv2.FILLED)

                if distancia <= radius:
                    cv2.putText(img, str(ok), (10, 90),
                                cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 3)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.circle(img, center_coordinates, radius, color, thickness)
    img = cv2.flip(img, 1)

    img = cv2.resize(img, (1920, 1080), fx=0, fy=0,
                     interpolation=cv2.INTER_CUBIC)
    cv2.putText(img, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
