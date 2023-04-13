from pickle import FRAME
import cv2 
import time 
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))
cap = cv2.VideoCapture(0)
time.sleep(2)
bg = 0

for i in range(60):
    ret,beg  = cap.read()
bg = np.flip(bg,axis=0)

while (cap.isOpened()):
    ret,img = cap.read()
    if not ret:
        break

    img = np.flip(img,axis=0)

frame = cv2.resize(FRAME, (640,480))
image = cv2.resize(img, (640,480))

hsv = cv2.cvtColor(img, cv2.COLOR_BRG2HSV)
u_black = np.array([104,153, 70])
l_black = np.array([30,30,0])
mask = cv2.inRange(frame,u_black,l_black)
res  = cv2.bitwise_and(frame, frame, mask=mask)

f = frame.res
f = np.where(f == 0 ,image, f)
cv2.imshow("magic",f)
cv2.waitKey(1)

cap.release()

cv2.destroyAllWindows()



