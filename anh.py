import cv2 
import numpy as np
#convert color
def convert_color(img, mode):
    img = cv2.imread(img)
    if mode == "hsv":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif mode == "gray":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif mode == "bin" :
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, img = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    return img

#erosion & dilation
def morphological_operations(img, mode):
    img = cv2.imread(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel =np.ones((5,5), np.uint8)
    if mode == "erode":
        img = cv2.erode(img_gray, kernel, iterations=1)
    elif mode == "dilate":
        img = cv2.dilate(img_gray, kernel, iterations=1)
    return img

#edge detection
def edge_detection(img, mode):
    img = cv2.imread(img)
    img = img[190:,:]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_blur = cv2.blur(img_gray, (3,3))
    if mode == "sobel_x":
        img = cv2.Sobel(image_blur, cv2.CV_64F, 0, 1, ksize=5)
    elif mode == "sobel_y":
        img = cv2.Sobel(image_blur, cv2.CV_64F, 1, 0, ksize=5)
    elif mode == "canny":
        img = cv2.Canny(image_blur, 50, 100)
    return img

#hough transform
def hough_transform(img, mode):
    img = cv2.imread(img)
    img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.blur(img_gray, (3,3))
    img_canny = cv2.Canny(img_blur, 50, 100)
    if mode == "houghline":
        lines = cv2.HoughLines(img_canny, 1, np.pi/180, 150, None, 0 ,0)
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0= b*rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv2.line(img, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)
        return img
    elif mode == "houghlineP":
        img_houghline_P = np.copy(img)
        linesP = cv2.HoughLinesP(img_canny, 1, np.pi/180, 50, None, 50, 10)
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv2.line(img_houghline_P, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv2.LINE_AA)
        return img_houghline_P

#contour
def contour(img):
    img = cv2.imread(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    return img

#lane detection with hough transform & contour
def lane_detech(img, mode):
    img = cv2.imread(img)
    if mode == "hough":
        img = img[130:,:]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
        canny = cv2.Canny(img_blur, 150, 255)

        _img_bgr = np.copy(img)
        linesP = cv2.HoughLinesP(canny, 1, np.pi/180, 50, None, 50, 10)
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv2.line(_img_bgr, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv2.LINE_AA)
        return _img_bgr
    if mode == "contour":
        img = img[130:,:]
        lower_green = np.array([95, 0, 0], dtype="uint8")
        upper_green = np.array([255, 255, 255], dtype="uint8")
        mask = cv2.inRange(img, lower_green, upper_green)
        lane = cv2.bitwise_and(img, img, mask= mask)
        lane_gray = cv2.cvtColor(lane, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(lane_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        area = []
        for i, contour in enumerate(contours):
            area.append(cv2.contourArea(contour))
        max_area = np.max(area)
        idx = area.index(max_area)
        cnt = contours[idx]
        cv2.drawContours(lane, [cnt], 0, [255,0,255],1)
        return lane
#Display image
#cv2.imshow("convert_color", convert_color('car_yellow_173.0.jpg', 'bin'))
#cv2.imshow("morphological_operations", morphological_operations('lane.jpg', 'erode'))

img = edge_detection('img_OD.jpg', 'canny')
print(img.shape)
# Define an array of endpoints of triangle
points = np.array([[0, 0], [0, 100], [180, 28], [460, 28],[640, 100], [640, 0]])
# Use fillPoly() function and give input as
# image, end points,color of polygon
# Here color of polygon will blue
cv2.fillPoly(img, pts=[points], color=(0, 0, 0))
cv2.imshow("edge detection",img)
# cv2.imwrite('roi.jpg', img)
# cv2.imwrite('canny.jpg', img)
#cv2.imshow("hough_transform", hough_transform('car_yellow_173.0.jpg', 'houghlineP'))
#cv2.imshow("contour", contour('mau-thiet-ke-bong-bong-vector-03.jpg'))
#cv2.imshow("lane detection", lane_detech('car_yellow_173.0.jpg', 'contour'))
cv2.waitKey()