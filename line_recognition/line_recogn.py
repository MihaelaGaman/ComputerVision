from scipy import ndimage
from scipy import misc
from scipy import signal

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import cv2
import math
import random
import copy

'''
Select S points random (uniformly)
'''
def select_samples(all_points, S):
	samples = []

	for iter in range(S):
		i = int(random.uniform(0, len(all_points)))

		while len(samples) > 0 and all_points[i] in samples:
			i = int(random.uniform(0, len(all_points)))

		samples.append(all_points[i])

	return samples

'''
Fit line to S points
Hough transform
S = 2
'''
def fit_line2(samples):
	m = 0
	b = 0

	n = len(samples)
	sx = 0
	sy = 0
	Sxy = 0
	Sx2 = 0

	# Compute x-es/y sum
	for (x, y) in samples:
		sx += x
		sy += y

	sx = sx / n
	sy = sy / n

	# Sxy = sum of( (x - sx)(y - sy))
	# Sx2 = sum of((x - sx)^2)
	for (x, y) in samples:
		Sxy += (x - sx) * (y - sy)
		Sx2 += (x - sx) * (x - sx)

	if Sx2 != 0:
		m = int(Sxy / Sx2)

	b = int(sy - m * sx)

	return m, b

'''
Compute distance as:
	d(mx - y + b = 0, (x0, y0)) 
	= abs(mx0 - y0 + b) / sqrt(m * m + 1)
'''
def distance(point, line):
	(x0, y0) = point
	(m, b) = line

	d = float(abs(m * x0 - y0 + b) / math.sqrt(m * m + 1))

	return d

''' Points whose distance < t
	from the line
'''
def find_inliers(all_points, line, t):
	inliers = []

	for point in all_points: 
		d = distance(point, line)
		if d < t:
			inliers.append(point)

	return inliers

def ransac(all_points, N, S, t, d):
	lines = []
	points = []
	for iter in range(N):
		samples = select_samples(all_points, S)
		m, b = fit_line2(samples)
		if m != 0 or b != 0:
			inliers = find_inliers(all_points, (m, b), t)

			no_inliers = len(inliers)
			if no_inliers >= d:
				# Accept line
				lines.append((m, b))
				points.append(samples)

	return points, lines


'''
Get the points on the edges.
'''
def get_white_points(edges):
	whites = []

	for l in range(len(edges)):
		for c in range(len(edges[0])):
			if edges[l][c] == 255:
				whites.append((l, c))

	return whites

'''
Draw lines in image
'''
def draw_lines(img, points, lines):
	res = copy.deepcopy(img)

	for k in range(len(points)):
		xes = [xi[0] for xi in points[k]]
		yces = [yi[1] for yi in points[k]]
		xmin = min(xes); xmax = max(xes)
		ymin = min(yces); ymax = max(yces)

		for l in range(xmin, xmax):
			for c in range(ymin, ymax):
				if distance((l, c), lines[k]) == 0:
					res[l][c] = [255, 0, 0]

	return res

def main():
	# image to be used
	file_img_name = 'urban4.jpg'

	f = misc.face()
	f = misc.imread(file_img_name)

	img = cv2.imread(file_img_name,0)

	edges = cv2.Canny(img,100,200)

	whites = get_white_points(edges)


	# Parameters
	N = 100
	S = 30
	t = 1
	d = 20

	points, lines = ransac(whites, N, S, t, d)
	f = draw_lines(f, points, lines)


	plt.imshow(f)
	plt.show()

	#gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	plt.subplot(121),plt.imshow(img)
	#plt.subplot(121),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(edges,cmap = 'gray')
	plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

	plt.show()


main()
