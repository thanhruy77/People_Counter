import numpy as np
import cv2 as cv
import Person
import time

def set_line(val):
    global line_down_x, line_down_y
    line_down_x = val[0]
    line_down_y = val[1]
    update_line()

def on_trackbar(val):
    pass

def update_line():
    global pts_L1
    pt1 = [0, line_down_y]
    pt2 = [width, line_down_y]
    pt3 = [line_down_x, 0]
    pt4 = [line_down_x, height]
    pts_L1 = np.array([pt1, pt2, pt3, pt4], np.int32)
    pts_L1 = pts_L1.reshape((-1, 1, 2))

try:
    log = open('log.txt', "w")
except:
    print("No se puede abrir el archivo log")

cnt_down = 0

cap = cv.VideoCapture('video.mp4')
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

for i in range(19):
    print(i, cap.get(i))

frameArea = height * width
areaTH = frameArea / 250
print('Area Threshold', areaTH)

line_down_x = int(width / 2)
line_down_y = int(height / 2)

print("Red line x:", str(line_down_x))
print("Red line y:", str(line_down_y))
line_down_color = (255, 0, 0)

pt1 = [0, line_down_y]
pt2 = [width, line_down_y]
pt3 = [line_down_x, 0]
pt4 = [line_down_x, height]
pts_L1 = np.array([pt1, pt2, pt3, pt4], np.int32)
pts_L1 = pts_L1.reshape((-1, 1, 2))

fgbg = cv.createBackgroundSubtractorMOG2(detectShadows=True)

kernelOp = np.ones((3, 3), np.uint8)
kernelCl = np.ones((11, 11), np.uint8)

font = cv.FONT_HERSHEY_SIMPLEX
persons = []
max_p_age = 5
pid = 1

cv.namedWindow('Frame')

# Tạo trackbar để điều chỉnh đường line theo bất kỳ hướng nào
cv.createTrackbar('Line', 'Frame', 0, 1, on_trackbar)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('EOF')
        print('DOWN:', cnt_down)
        break

    for i in persons:
        i.age_one()

    fgmask = fgbg.apply(frame)

    ret, imBin = cv.threshold(fgmask, 200, 255, cv.THRESH_BINARY)
    mask = cv.morphologyEx(imBin, cv.MORPH_OPEN, kernelOp)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernelCl)

    contours0, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours0:
        area = cv.contourArea(cnt)
        if area > areaTH:
            M = cv.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            x, y, w, h = cv.boundingRect(cnt)

            new = True
            for i in persons:
                if abs(x - i.getX()) <= w and abs(y - i.getY()) <= h:
                    new = False
                    i.updateCoords(cx, cy)
                    if i.going_DOWN(line_down_y, line_down_x):
                        cnt_down += 1
                        print("ID:", i.getId(), 'crossed going down at', time.strftime("%c"))
                        log.write("ID: " + str(i.getId()) + ' crossed going down at ' + time.strftime("%c") + '\n')
                    break
                if i.getState() == '1' and i.getDir() == 'down':
                    i.setDone()
                if i.timedOut():
                    index = persons.index(i)
                    persons.pop(index)
                    del i
            if new:
                p = Person.MyPerson(pid, cx, cy, max_p_age)
                persons.append(p)
                pid += 1

            cv.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            img = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for i in persons:
        cv.putText(frame, str(i.getId()), (i.getX(), i.getY()), font, 0.3, i.getRGB(), 1, cv.LINE_AA)

    str_down = 'DOWN: ' + str(cnt_down)
    frame = cv.polylines(frame, [pts_L1], False, line_down_color, thickness=2)
    cv.putText(frame, str_down, (10, 90), font, 0.5, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(frame, str_down, (10, 90), font, 0.5, (255, 0, 0), 1, cv.LINE_AA)

    cv.imshow('Frame', frame)
    # cv.imshow('Mask', mask)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

log.flush()
log.close()
cap.release()
cv.destroyAllWindows()