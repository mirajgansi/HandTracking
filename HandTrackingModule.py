import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, 
                                       min_detection_confidence=self.detectionCon,
                                       min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        # Draw landmarks if hands are detected
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # Draw the landmarks and connections
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            # Check if the requested hand exists
            if len(self.results.multi_hand_landmarks) > handNo:
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    if draw :  # Draw circle on first landmark
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return lmList

#def fingersUp()

def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    pTime = 0
    
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            break
            
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        
        if len(lmList) != 0:
            # Example: Print position of landmark 8 (index finger tip)
            print(lmList[4])
        
        # FPS calculation
        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime > pTime else 0
        pTime = cTime
        
        # Display FPS on the image
        cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 
                   1, (255, 0, 255), 2)
        
        # Show the image with landmarks and FPS
        cv2.imshow("Image", img)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()