import cv2
from tracker import *

tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("DSC_0218.MOV")

object_detector= cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()

    h,w,_ = frame.shape
    
    # h=1080, w = 1920
    #region of interest roi
    roi = frame[350:900,700:1300]




    mask = object_detector.apply(roi)
    _,mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for i in contours:
        #first we find the are of the mask and then we see if the area is greater than 100 pixels only then mask it
        area = cv2.contourArea(i)
        if area > 12000:
            #cv2.drawContours(roi, [i], -1 , (0,255,0), 2 )
            x,y,wi,he= cv2.boundingRect(i)
            cv2.rectangle(roi, (x,y), (x+wi, y+he),(0,255,0),3)
            detections.append([x,y,wi,he])



    #tracking
    box_IDs = tracker.update(detections)
    for box_ID in box_IDs:
        x,y,wi,he, id = box_ID
        cv2.putText(roi, str(id), (x,y-15), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 2)

    #cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    #cv2.imshow("Mask", mask)

    key = cv2.waitKey(27)
    if(key==27):
        break;
cap.release()
cv2.destroyAllWindows()