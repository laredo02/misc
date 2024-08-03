
import cv2 as cv

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Unable to open video capture device.")
    exit()

while True:
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)    
    
    if ret:
        cv.imshow('Webcam Video', frame)
        if cv.waitKey(1) & 0xff == ord('q'):
            break
    else:
        print("ERROR: Unable to read frame.")
        break
    




