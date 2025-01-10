import os
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from tkinter import Tk, filedialog

# Variables
width, height = 1280, 720
folderPath = "Presentation"

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Presentation images
pathImages = sorted(os.listdir(folderPath), key=len)
print(pathImages)

# Select video dynamically
Tk().withdraw()  # Hides the root tkinter window
videoPath = filedialog.askopenfilename(
    title="Select a Video File",
    filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov")]
)

if not videoPath:
    print("No video selected. Exiting.")
    exit()

# Video setup
capVideo = cv2.VideoCapture(videoPath)
if not capVideo.isOpened():
    print("Error: Unable to open the video file. Check the selected file.")
    exit()

videoPlaying = False
videoPaused = False

# Variables
imgNumber = 0
hs, ws = 130, 213
gestureThreshold = 300
buttonPressed = False
buttonCounter = 0
buttonDelay = 30
annotations = [[]]
annotationNumber = -1
annotationStart = False

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

while True:
    # Import images
    success, img = cap.read()
    img = cv2.flip(img, 1)
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)

    hands, img = detector.findHands(img)
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 2)

    if hands and buttonPressed is False:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand['center']
        lmList = hand['lmList']

        # Gesture logic
        xVal = int(np.interp(lmList[8][0], [width // 2, width], [0, width]))
        yVal = int(np.interp(lmList[8][1], [100, height - 100], [0, height]))
        indexFinger = xVal, yVal

        if cy <= gestureThreshold:
            # Navigate slides
            if fingers == [1, 0, 0, 0, 0]:  # Left
                if imgNumber > 0:
                    buttonPressed = True
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False
                    imgNumber -= 1
            if fingers == [0, 0, 0, 0, 1]:  # Right
                if imgNumber < len(pathImages) - 1:
                    buttonPressed = True
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False
                    imgNumber += 1

            # Control video
            if fingers == [0, 1, 0, 0, 0]:  # Play/Pause
                if videoPlaying:
                    videoPaused = not videoPaused
                else:
                    videoPlaying = True
                buttonPressed = True
            if fingers == [0, 1, 1, 0, 0] and videoPlaying:  # Seek Forward
                currentFrame = capVideo.get(cv2.CAP_PROP_POS_FRAMES)
                capVideo.set(cv2.CAP_PROP_POS_FRAMES, currentFrame + 30)
                buttonPressed = True
            if fingers == [0, 0, 1, 1, 0] and videoPlaying:  # Seek Backward
                currentFrame = capVideo.get(cv2.CAP_PROP_POS_FRAMES)
                capVideo.set(cv2.CAP_PROP_POS_FRAMES, max(currentFrame - 30, 0))
                buttonPressed = True

    # Button press delay
    if buttonPressed:
        buttonCounter += 1
        if buttonCounter > buttonDelay:
            buttonCounter = 0
            buttonPressed = False

    # Display video
    if videoPlaying and not videoPaused:
        successVideo, imgVideo = capVideo.read()
        if successVideo:
            h, w, _ = imgCurrent.shape
            imgVideoResized = cv2.resize(imgVideo, (ws, hs))
            imgCurrent[0:hs, w - ws:w] = imgVideoResized
        else:
            print("Video playback finished.")
            videoPlaying = False

    # Add webcam to slides
    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurrent.shape
    imgCurrent[0:hs, w - ws:w] = imgSmall

    # Draw annotations
    for i in range(len(annotations)):
        for j in range(len(annotations[i])):
            if j != 0:
                cv2.line(imgCurrent, annotations[i][j - 1], annotations[i][j], (0, 0, 200), 5)

    # Show images
    cv2.imshow("Webcam", img)
    cv2.imshow("Presentation", imgCurrent)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
capVideo.release()
cv2.destroyAllWindows()
