import math
import cv2
import operator
import numpy as np

def distance2(a,b):
	blue = (a[0] - b[0])
	green = (a[1] - b[1])
	red = (a[2] - b[2])
	d = (red**2 + green**2 + blue**2)/3
	return d

def distance5d(a,b):
	p = [2,2,2,2,2]
	d = math.sqrt(reduce(operator.add,map(operator.pow,map(operator.sub,a,b),p)))
	return d

# Show image
def showImg(img):
	cv2.imshow("image", img) 
	k = cv2.waitKey(0) & 0xFF
	if k == 27:         # wait for ESC key to exit
	    cv2.destroyAllWindows()

def gaussian(x,std):
	return (np.exp(-((x)**2)/(2*std*std)))/(math.sqrt(math.pi*2)*std)

def GBlur(img, kernel_radius, st_dev):
	kernel_diameter = kernel_radius*2
	kernel_size =  kernel_diameter + 1
	asArray = np.asarray
	cols = len(img[0])
	lines = len(img)
	img_ = img.tolist()
	matrix = img.tolist()
	ret_matrix = img.tolist()
	#stddev2pi = st_dev*st_dev*sqrt2pi**2
	K = range(kernel_size)
	weights = [None]*kernel_size
	gauss = gaussian
	sum_w = 0.0
	for k in K:
		x = k-kernel_radius
		gauss_w = gauss(x,st_dev)
		weights[k] = float(gauss_w)
		sum_w += gauss_w
	W = (asArray(weights)/sum_w).tolist()
	L = range(lines)
	C = range(cols)
	for i in L:
		for j in C:
			p = 0
			u = j-kernel_size
			A = img_[i][u:j]
			if(u<0):
				A = list(img_[i][u:] + img_[i][:j])
			else:
				A = list(img_[i][u:j])
			matrix[i][j-kernel_radius] = reduce(operator.add, map(operator.mul,W,A))
	matrix = (asArray(matrix).transpose()).tolist()
	for j in C:
		for i in L:
			p = 0
			u = i-kernel_size
			A = []
			if(u<0):
				A = list(matrix[j][u:] + matrix[j][:i])
			else:
				A = list(matrix[j][u:i])
			ret_matrix[i-kernel_radius][j] = reduce(operator.add, map(operator.mul,W,A))
	return asArray(ret_matrix, np.uint8)

def sum5d(a,b):
	return map(operator.add,a,b)
