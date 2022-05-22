import random
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
f = open("dados.txt", "a")

def drawCircle(img):
    cv2.circle(img, center_coordinates, radius, color, thickness)


x = random.randint(0, 640)
y = random.randint(0, 480)
center_coordinates = (x, y)
radius = 15
color = (0, 0, 255)
thickness = -1
frameCounter = 100
ok = "ok"
contCirc = 0
inicio = "Encoste no circulo para iniciar"
startState = 0
tempoInicial = 0
tempoRest = 20
tempoRestAux = 0
fimCont = 0
zero = 0
sendOk = 0
fileWrite = 0
while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

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

                if fimCont != 1:
                    if distancia <= radius:

                        distancia2 = math.sqrt((x-cx)**2 + (y-cy)**2)

                        while (distancia2 < 250):
                            x = random.randint(0, 640)
                            y = random.randint(0, 480)
                            center_coordinates = (x, y)
                            distancia2 = math.sqrt((x-cx)**2 + (y-cy)**2)


                        frameCounter = 0
                        if startState == 1:
                            contCirc = contCirc + 1
                        startState = 1
                 

               

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    drawCircle(img)

    img = cv2.flip(img, 1)

    img = cv2.resize(img, (1920, 1080), fx=0, fy=0,
                     interpolation=cv2.INTER_CUBIC)

    cv2.putText(img, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.putText(img, str(int(contCirc)), (10, 150),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)


    if frameCounter < 10:

        cv2.putText(img, str(ok), (1600, 120), cv2.FONT_HERSHEY_PLAIN, 15, (255, 255, 0), 3)
        frameCounter = frameCounter + 1    

    if startState == 0:

        cv2.putText(img, str(inicio), (640, 230),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        tempoInicial = time.time()
        tempoRestAux = tempoRest
        
    tempoAux = time.time()
    tempoDecorrido = tempoAux - tempoInicial
    
    if tempoInicial !=  0 and fimCont != 1:
        tempoRestAux = tempoRestAux - tempoDecorrido
        cv2.putText(img, str(int(tempoRestAux)), (960, 90),
                cv2.FONT_HERSHEY_PLAIN, 7, (0, 165, 255), 3)
        if tempoRestAux < 1 :
            fimCont = 1
        tempoRestAux = tempoRest
    if fimCont == 1:
        cv2.putText(img, str(int(zero)), (960, 90),
                cv2.FONT_HERSHEY_PLAIN, 7, (0, 165, 255), 3)
        contCircS = str(contCirc)        
        
    if fimCont == 1 and fileWrite == 0:    
        f.write("\n" + contCircS) 
        fileWrite = 1  
        f.close()    
        

    cv2.imshow("Image", img)
    cv2.waitKey(1)
