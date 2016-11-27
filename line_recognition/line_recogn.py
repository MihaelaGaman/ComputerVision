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

# Select S points random (uniformly)
def select_sample(all_points):
	i1 = int(random.uniform(0, len(all_points)))
	i2 = int(random.uniform(0, len(all_points)))

	while i1 == i2:
		i2 = int(np.random.uniform(0, len(all_points)))

	p1 = all_points[i1]
	p2 = all_points[i2]

	return p1, p2

# Fit line to S points
# Hough transform
# S = 2
def fit_line2(p1, p2):
	m = 0
	b = 0
	print p1, p2
	(x0, y0) = p1
	(x1, y1) = p2

	if abs(x1 - x0) != 0:
		m = int(abs(y1 - y0) / abs(x1 - x0))
		b = int(y0 - x0 * abs(y1 - y0) / abs(x1 - x0))

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
		p1, p2 = select_sample(all_points)
		m, b = fit_line2(p1, p2)
		if m != 0 or b != 0:
			inliers = find_inliers(all_points, (m, b), t)

			no_inliers = len(inliers)
			if no_inliers >= d:
				# Accept line
				lines.append((m, b))
				points.append((p1, p2))

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
		(p1, p2) = points[k]
		xmin = min(p1[0], p2[0]); xmax = max(p1[0], p2[0])
		ymin = min(p1[1], p2[1]); ymax = max(p1[1], p2[1])

		for l in range(xmin, xmax):
			for c in range(ymin, ymax):
				if distance((l, c), lines[k]) == 0:
					res[l][c] = [255, 0, 0]

	return res

def main():
	# image to be used
	file_img_name = 'forrest.jpg'

	f = misc.face()
	f = misc.imread(file_img_name)

	img = cv2.imread(file_img_name,0)

	edges = cv2.Canny(img,100,200)

	whites = get_white_points(edges)


	# Parameters
	N = 300
	S = 2
	t = 2
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
