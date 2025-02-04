import os
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

width, height = 1280, 720
folderPath = "Presentation"
videoPath = "video.mp4"  

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

capVideo = cv2.VideoCapture(videoPath)

pathImages = sorted(os.listdir(folderPath), key=len)

# Variables
imgNumber = 0
hs, ws = 130, 213
videoWidth, videoHeight = 540, 260  
gestureThreshold = 300
buttonPressed = False
buttonCounter = 0
buttonDelay = 30
annotations = [[]]
annotationNumber = -1
annotationStart = False
paused = False
mute = False
volume = 50  

detector = HandDetector(detectionCon=0.8, maxHands=1)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)

    if not paused:
        success, frame = capVideo.read()
        if not success:
            capVideo.set(cv2.CAP_PROP_POS_FRAMES, 0) 
            continue

    frame = cv2.resize(frame, (videoWidth, videoHeight))

    hands, img = detector.findHands(img)
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 2)

    if hands and buttonPressed is False:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand['center']
        lmList = hand['lmList']

        xVal = int(np.interp(lmList[8][0], [width // 2, width], [0, width]))
        yVal = int(np.interp(lmList[8][1], [100, height - 100], [0, height]))
        indexFinger = xVal, yVal

        if cy <= gestureThreshold:  
            if fingers == [1, 0, 0, 0, 0]:  # Previous Slide
                if imgNumber > 0:
                    buttonPressed = True
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False
                    imgNumber -= 1
            if fingers == [0, 0, 0, 0, 1]:  # Next Slide
                if imgNumber < len(pathImages) - 1:
                    buttonPressed = True
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False
                    imgNumber += 1

        if fingers == [0, 1, 1, 0, 0]:  # Pointer
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
        if fingers == [0, 1, 0, 0, 0]:  # Draw
            if annotationStart is False:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
            annotations[annotationNumber].append(indexFinger)
        else:
            annotationStart = False
        if fingers == [0, 1, 1, 1, 0]:  # Erase Last Annotation
            if annotations:
                annotations.pop(-1)
                annotationNumber -= 1
                buttonPressed = True

        if fingers == [0, 1, 1, 0, 0]:  # Pause/Resume
            paused = not paused
            print("Pause/Resume Video")
        if fingers == [1, 1, 1, 1, 1]:  # Forward 10s
            capVideo.set(cv2.CAP_PROP_POS_MSEC, capVideo.get(cv2.CAP_PROP_POS_MSEC) + 10000)
            print("Forward 10s")
        if fingers == [1, 1, 0, 0, 0]:  # Backward 10s
            capVideo.set(cv2.CAP_PROP_POS_MSEC, max(capVideo.get(cv2.CAP_PROP_POS_MSEC) - 10000, 0))
            print("Backward 10s")
        if fingers == [1, 0, 0, 0, 1]:  # Mute/Unmute
            mute = not mute
            print("Mute" if mute else "Unmute")
        if fingers == [1, 0, 0, 0, 0]:  # Increase Volume
            volume = min(volume + 10, 100)
            print(f"Increase Volume: {volume}")
        if fingers == [0, 0, 0, 0, 1]:  # Decrease Volume
            volume = max(volume - 10, 0)
            print(f"Decrease Volume: {volume}")

    if buttonPressed:
        buttonCounter += 1
        if buttonCounter > buttonDelay:
            buttonCounter = 0
            buttonPressed = False

    for i in range(len(annotations)):
        for j in range(len(annotations[i])):
            if j != 0:
                cv2.line(imgCurrent, annotations[i][j - 1], annotations[i][j], (0, 0, 200), 5)

    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurrent.shape
    imgCurrent[0:hs, w - ws:w] = imgSmall

    img[10:10+videoHeight, 10:10+videoWidth] = frame

    cv2.imshow("Webcam Feed", img)
    cv2.imshow("Slides", imgCurrent)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
capVideo.release()
cv2.destroyAllWindows()
