# coding=utf-8
# 导入相应的pthon包
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

# 计算中心点函数
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
i = 0

def paint_calculate(ptA, color):
	xA = ptA[0]
	yA = ptA[1]
	cv2.circle(orig, (int(xA), int(yA)), 2, color, -1)
	cv2.circle(orig, (int(0.5 * w), int(0.5 * h)), 2, color, -1)
	cv2.circle(orig, (int(w), int(h)), 3, color, -1)
	cv2.line(orig, (int(xA), int(yA)), (int(0.5 * w), int(0.5 * h)), color, 1)
	cv2.line(orig, (int(w), int(h)), (int(0.5 * w), int(0.5 * h)), color, 1)

	# 计算坐标之间的欧式距离并及进行距离转换
	D1 = dist.euclidean((xA, yA), (int(0.5 * w), int(0.5 * h))) / refObj[2]
	(mX, mY) = midpoint((xA, yA), (int(0.5 * w), int(0.5 * h)))
	cv2.putText(orig, "{:.3}".format(D1), (int(mX), int(mY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

	D2 = dist.euclidean((int(w), int(h)), (int(0.5 * w), int(0.5 * h))) / refObj[2]
	(mX, mY) = midpoint((int(w), int(h)), (int(0.5 * w), int(0.5 * h)))
	cv2.putText(orig, "{:.3f}".format(D2), (int(mX), int(mY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
	cv2.putText(orig, "RATE:{:.4f}".format(D1 / D2), (int(mX - 300), int(mY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
				color, 1)
	# 显示结果

	cv2.imshow("Image", orig)
	cv2.waitKey(0)



# 进行参数配置和解析
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

# 读取图片
image = cv2.imread(args["image"])

# 执行灰度变换
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 执行高斯滤波
gray = cv2.GaussianBlur(gray, (7, 7), 0)

h = image.shape[0]
w = image.shape[1]

# 执行Canny边缘检测
edged = cv2.Canny(gray, 50, 100)
# 执行腐蚀和膨胀后处理
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# 在边缘映射中寻找轮廓
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# 对轮廓点进行排序
(cnts, _) = contours.sort_contours(cnts)
# 设置显示颜色

refObj = None

# 循环遍历每一个轮廓点
for c in cnts:
	# 过滤点太小的轮廓点
	if cv2.contourArea(c) < 100:
		continue

	# 计算最小的外接矩形
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")

	# 对轮廓点进行排序
	box = perspective.order_points(box)

	# 计算BB的中心点
	cX = np.average(box[:, 0])
	cY = np.average(box[:, 1])

	if refObj is None:
		# 获取4个坐标点并计算中心点坐标
		(tl, tr, br, bl) = box
		(tlblX, tlblY) = midpoint(tl, bl)
		(trbrX, trbrY) = midpoint(tr, br)

		# 计算中心点之间的欧式距离
		D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
		# 获取计算结果
		# refObj = (box, (cX, cY), D / args["width"])
		refObj = (box, (cX, cY), D / 1.0)


	# 绘制轮廓
	orig = image.copy()
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
	cv2.drawContours(orig, [refObj[0].astype("int")], -1, (0, 255, 0), 2)

	# 进行坐标堆叠
	refCoords = np.vstack([refObj[0], refObj[1]])
	objCoords = np.vstack([box, (cX, cY)])

	# 遍历所有的坐标点

	xN = (refCoords[0][0] + refCoords[1][0]) * 0.5
	yN = (refCoords[0][1] + refCoords[1][1]) * 0.5

	xE = (refCoords[1][0] + refCoords[2][0]) * 0.5
	yE = (refCoords[1][1] + refCoords[2][1]) * 0.5

	xS = (refCoords[2][0] + refCoords[3][0]) * 0.5
	yS = (refCoords[2][1] + refCoords[3][1]) * 0.5

	xW = (refCoords[3][0] + refCoords[0][0]) * 0.5
	yW = (refCoords[3][1] + refCoords[0][1]) * 0.5

	xC = refCoords[4][0]
	yC = refCoords[4][1]

	colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0), (255, 0, 255))

		# 绘制点并连接为直线
	paint_calculate((xN, yN), colors[0])
	orig = image.copy()
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
	cv2.drawContours(orig, [refObj[0].astype("int")], -1, (0, 255, 0), 2)

	paint_calculate((xE, yE), colors[1])
	orig = image.copy()
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
	cv2.drawContours(orig, [refObj[0].astype("int")], -1, (0, 255, 0), 2)

	paint_calculate((xW, yW), colors[2])
	orig = image.copy()
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
	cv2.drawContours(orig, [refObj[0].astype("int")], -1, (0, 255, 0), 2)

	paint_calculate((xS, yS), colors[3])
	orig = image.copy()
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
	cv2.drawContours(orig, [refObj[0].astype("int")], -1, (0, 255, 0), 2)

	paint_calculate((xC, yC), colors[4])

