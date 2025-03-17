import cv2
import os
import time
import HandTrackingModule as htm

# Set camera resolution
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Load overlay images
folderPath = "image"
myList = os.listdir(folderPath)
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    if image is None:
        print(f"Error loading image: {imPath}")
        continue
    image = cv2.resize(image, (200, 200))  # Resize to 200x200 for consistency
    overlayList.append(image)

print(f"Loaded {len(overlayList)} images")

pTime = 0
detector = htm.handDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]

# Main loop
while True:
    success, img = cap.read()
    if not success:
        break  # If no frame is captured, break the loop

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if lmList:
        fingers = []

        # Thumb detection (assumes right-hand detection)
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:  
            fingers.append(1)
        else:
            fingers.append(0)

        # Other four fingers
        for id in range(1, 5):  # Start from index 1 (skip thumb)
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)
        print(f"Fingers: {totalFingers}")

        # Overlay Image based on fingers count
        if 0 <= totalFingers <= len(overlayList):
            img[0:200, 0:200] = overlayList[totalFingers - 1]

        cv2.rectangle(img,(20,225),(170,425),(0,255,0),cv2.FILLED)
        cv2.putText(img,str(totalFingers),(45,375),cv2.FONT_HERSHEY_COMPLEX,5,(255,0,0),25)
    # FPS Calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime) if pTime != 0 else 0  # Avoid division by zero
    pTime = cTime

    # Display FPS on the frame
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

    # Show frame
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
