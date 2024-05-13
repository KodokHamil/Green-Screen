import cv2 as cv
import numpy as np

cam = cv.VideoCapture(0)
image = cv.imread("Tower2.jpg")

if not cam.isOpened():
    print("error opening camera")
    exit()
    
def nothing(x):
    pass    
cv.namedWindow("HSV bar")
cv.resizeWindow("HSV bar",300,300)    

cv.createTrackbar("L-H", "HSV bar", 0, 179, nothing)
cv.createTrackbar("L-S", "HSV bar", 0, 255, nothing) 
cv.createTrackbar("L-V", "HSV bar", 0, 255, nothing)
cv.createTrackbar("U-H", "HSV bar", 179, 179, nothing)
cv.createTrackbar("U-S", "HSV bar", 255, 255, nothing)
cv.createTrackbar("U-V", "HSV bar", 255, 255, nothing)

    
while True:
    ret, frame = cam.read()
    
    frame = cv.resize(frame, (640, 480))
    image = cv.resize(image, (640, 480))
    
    hsv=cv.cvtColor(frame, cv.COLOR_BGR2HSV)    #optimalnya:
    l_h = cv.getTrackbarPos("L-H", "HSV bar")   #0
    l_s = cv.getTrackbarPos("L-S", "HSV bar")   #0
    l_v = cv.getTrackbarPos("L-V", "HSV bar")   #178
    u_h = cv.getTrackbarPos("U-H", "HSV bar")   #179
    u_s = cv.getTrackbarPos("U-S", "HSV bar")   #36
    u_v = cv.getTrackbarPos("U-V", "HSV bar")   #255
    l_green = np.array([l_h, l_s, l_v])
    u_green = np.array([u_h, u_s, u_v])

    
    mask = cv.inRange(hsv, l_green, u_green)
    res = cv.bitwise_and(frame, frame, mask = mask)
    
    f = frame - res
    
    f = np.where(f==0, image, f)
    
    cv.imshow("cam", frame)
    cv.imshow("mask", f)
    if cv.waitKey(1) == ord('q'):
        break
cam.release()
cv.destroyAllWindows()