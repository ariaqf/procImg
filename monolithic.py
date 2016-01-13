#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import math
import operator
import array
import random
import time
import scipy.cluster.vq as km
from scipy import ndimage as nd
import cProfile
import gc

def distance2(a,b):
	if(len(a) < 3):
		return (a-b)**2
	else:
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

def abstract(img,K=100, M_tol=0.5):
	start_time = time.time()
	a = len(img)
	b = len(img[0])
	p_size = a*b
	points = []
	scale = 3.0
	distance = distance5d
	for i in range(a):
		for j in range(b):
			points.append([scale*i,scale*j] + (img[i,j]).tolist())
	clusters = []
	for k in range(K):
		clusters.append([])
	#Use the scipy kmeans
	cluster_centers = km.kmeans(np.asarray(points),K)[0]
	for i in points:
		myDistance = float("inf")
		myCluster = 0
		for k in range(len(cluster_centers)):
			d = distance(i,cluster_centers[k])
			if(d < myDistance):
				myDistance = d
				myCluster = k
		clusters[myCluster].append(i)
	
	retr = np.zeros((a,b,3),np.uint8)
	for i in range(len(cluster_centers)):
		l = len(clusters[i])
		for j in range(l):
			retr[clusters[i][j][0]/scale,clusters[i][j][1]/scale] = np.asarray(cluster_centers[i][2:])
	return retr

def myabstract(img,K=100, M_tol=0.5):
	start_time = time.time()
	a = len(img)
	b = len(img[0])
	p_size = a*b
	points = []
	scale = 3.0
	distance = distance5d
	for i in range(a):
		for j in range(b):
			points.append([scale*i,scale*j] + (img[i,j]).tolist())
	clusters = []
	for k in range(K):
		clusters.append([])
	#My kmeans may be better
	cluster_centers = random.sample(points,K)
	rPS = range(p_size)
	rK = range(K)
	past_cluster_centers = []
	tol = 5
	l_tol = 0
	d_tol = (math.sqrt((tol-l_tol)**2))
	tries=0
	while((d_tol > M_tol) and (tries < 50)):
		for i in rK:
			clusters[i][:] = []
		for i in rPS:
			myCluster = 0
			myDistance = float("inf")
			for j in rK:
				dist = distance(cluster_centers[j],points[i])
				if(dist < myDistance):
					myDistance = dist
					myCluster = j
			clusters[myCluster].append(points[i][:])
		past_cluster_centers = cluster_centers[:]
		for i in rK:
			l = len(clusters[i])
			cluster_centers[i] = map(operator.div,reduce(sum5d,clusters[i]),[l]*5)
		l_tol = tol
		tol = reduce(operator.add,map(distance,cluster_centers,past_cluster_centers))
		d_tol = (math.sqrt((tol-l_tol)**2))
		print(tries,d_tol)
		tries+=1
	print("Average time per try:",(time.time()-start_time)/tries)
	for i in range(len(cluster_centers)):
		l = len(clusters[i])
		for j in range(l):
			retr[clusters[i][j][0]/scale,clusters[i][j][1]/scale] = np.asarray(cluster_centers[i][2:])
	return retr

def rarity(img,number=0.25):
	a = len(img)
	b = len(img[0])
	n_pixels = a*b
	start_time = time.time()
	
	small_image = cv2.cvtColor(np.float32(img/255.0),cv2.COLOR_BGR2LAB)
	small_image = small_image/1.0;

	#blur the image my son, we will use it later for weights
	w=b-1+(b%2)
	h=a-1+(a%2)
	blur_image2 = np.zeros((a,b), np.double)
	#blur_image = cv2.resize(small_image, (b,a))
	#std = 60.0
	#trunc = math.sqrt(n_pixels)/std
	for i in range(a):
		for j in range(b):
			p = small_image[i,j]/1.0;
			blur_image2[i,j] = np.dot(p,p)
	#blur_image = nd.filters.gaussian_filter(small_image, std, mode='constant', cval=0.0, truncate=trunc) 
	blur_image = cv2.GaussianBlur(small_image, (w,h), number*math.sqrt(n_pixels), borderType=cv2.BORDER_REPLICATE)
	#blur_image2 = nd.filters.gaussian_filter(blur_image2, std, mode='constant', cval=0.0, truncate=trunc) 
	blur_image2 = cv2.GaussianBlur(blur_image2, (w,h), number*math.sqrt(n_pixels), borderType=cv2.BORDER_REPLICATE)
	#channels = []
	#for c in cv2.split(img):
	#	channels.append(GBlur(c,min(a,b)/2 -1,25))
	#blur_image = cv2.merge(channels)

	#did you blur it?

	cj = np.zeros(3, np.double) # sum(b_j)
	cj2 = 0 # sum(b_j²)

	rarityMapFake = np.zeros((a,b), np.double)
	rarityMap = np.zeros((a,b), np.uint8)

	maxR = 0
	minR = 256*256
	for i in range(a):
		for j in range(b):
			p = small_image[i,j]/1.0
			r = np.dot(p,p) + blur_image2[i,j] - 2*np.dot(p,blur_image[i,j]) # sum(||a_i-b_j||²) = a_i²*sum(1) - 2*a_i*sum(b_j) + sum(b_j²)
			#r = r*255
			rarityMapFake[i,j] = r
			if(r < minR):
				minR = r
			if(r > maxR):
				maxR = r
	
	"""for i in range(a):
		for j in range(b):
			rarityMap[i,j] = np.uint8(255*(rarityMapFake[i,j])/(maxR))"""
	
	bigMap = 255*(rarityMapFake-minR)/(maxR-minR)
	return bigMap

def distribution(img, std=20, scale=2):
	A = len(img)
	B = len(img[0])
	a = A/scale
	b = B/scale
	n_pixels = a*b
	start_time = time.time()
	small_image = cv2.cvtColor(cv2.resize(img, (b,a)),cv2.COLOR_BGR2LAB)
	bigMap = np.zeros((a,b), np.double).tolist()
	meanMap = np.zeros((a,b,2), np.double).tolist()
	squaremeanMap = np.zeros((a,b,2), np.double).tolist()
	#meanPositionMap = np.zeros((,2), np.double)
	#Para uma dada cor i:
	#D_i = Sum(||P_j - mi_i||² * w(i,j)) 
	alpha = (2*std*std)
	beta = math.sqrt(math.pi*2)*std
	maxMap = 0.0

	for i in range(a):
		for j in range(b):
			p = small_image[i,j]/1.0
			sum_w = 0.0
			for i1 in range(a):
				for j1 in range(b):
					p_p1 = small_image[i1,j1]/1.0 - p
					x = np.dot(p_p1,p_p1)/alpha
					w = (math.e**(-x))/(beta)
					squaremeanMap[i][j][0] += w*i1*i1
					squaremeanMap[i][j][1] += w*j1*j1
					meanMap[i][j][0] += w*i1
					meanMap[i][j][1] += w*j1
					sum_w += w
			meanMap[i][j][0] /= sum_w
			meanMap[i][j][1] /= sum_w
			squaremeanMap[i][j][0] /= sum_w
			squaremeanMap[i][j][1] /= sum_w

	for i in range(a):
		for j in range(b):
			p = small_image[i,j]/1.0
			sum_w = 0
			for i1 in range(a):
				for j1 in range(b):
					p_p1 = small_image[i1,j1]/1.0 - p
					x = np.dot(p_p1,p_p1)/alpha
					w = (math.e**(-x))/(beta)
					bigMap[i][j] += w*((i1 - meanMap[i][j][0])**2 + (j1 - meanMap[i][j][1])**2)
					sum_w += w
			bigMap[i][j] /= sum_w
			if bigMap[i][j] > maxMap:
				maxMap = bigMap[i][j]
	for i in range(a):
		for j in range(b):
			bigMap[i][j] = 255*bigMap[i][j]/maxMap
	return cv2.resize(np.asarray(bigMap), (B,A))

def distribution2(img,std = 20, scale = 2, kernel_size = 253):
	totTime = time.time()
	gc.enable()
	A = len(img)
	B = len(img[0])
	a = A/scale
	b = B/scale
	n_pixels = a*b
	start_time = time.time()
	small_image = cv2.cvtColor(np.float32(img/255.0),cv2.COLOR_BGR2LAB)
	midMapi = np.zeros((256,256,256), np.double)
	midMapj = np.zeros((256,256,256), np.double)
	m2_ = np.zeros((256,256,256), np.double)
	w = np.zeros((256,256,256), np.double)
	bigMap = np.zeros((a,b), np.double)
	asArray = np.asarray
	for i in range(a):
		for j in range(b):
			l_ = int(small_image[i][j][0])
			a_ = int(small_image[i][j][1])
			b_ = int(small_image[i][j][2])
			v = np.zeros(2, np.double)
			v[0] = i
			v[1] = j
			midMapi[l_][a_][b_] += i
			midMapj[l_][a_][b_] += j
			m2_[l_][a_][b_] += np.dot(v,v)
			w[l_][a_][b_] += 1
	start_time = time.time()
	m1 = np.zeros((256,256,256,2), np.double)
	#m1[:,:,:,0] = nd.filters.gaussian_filter(midMapi, std, mode='wrap', cval=0.0)
	#m1[:,:,:,1] = nd.filters.gaussian_filter(midMapj, std, mode='wrap', cval=0.0)
	#m1_ = nd.filters.gaussian_filter(m2_, std, mode='wrap', cval=0.0)
	#better commands for scipy > 1.9
	trunc = (len(midMapi)/std)
	m1[:,:,:,0] = nd.filters.gaussian_filter(midMapi, std, mode='constant', cval=0.0, truncate=trunc)
	m1[:,:,:,1] = nd.filters.gaussian_filter(midMapj, std, mode='constant', cval=0.0, truncate=trunc)
	m1_ = nd.filters.gaussian_filter(m2_, std, mode='constant', cval=0.0, truncate=trunc)
	w = nd.filters.gaussian_filter(w, std, mode='constant', cval=0.0, truncate=trunc)
	maxMap = 0
	minMap = 0
	#m1 = asArray(m1)
	#m1_ = asArray(m1_)
	gc.collect()
	for i in range(a):
		for j in range(b):
			l_ = int(small_image[i][j][0])
			a_ = int(small_image[i][j][1])
			b_ = int(small_image[i][j][2])
			mean = np.dot(m1[l_,a_,b_],m1[l_,a_,b_])
			meanSquare = m1_[l_,a_,b_]
			bigMap[i][j] = (meanSquare-mean/w[l_,a_,b_])/w[l_,a_,b_]
			minMap = min(minMap, bigMap[i][j])
			maxMap = max(maxMap,bigMap[i][j])
	"""for i in range(a):
		for j in range(b):
			bigMap[i][j] = np.uint8(255*((bigMap[i][j]-minMap)/(maxMap-minMap)))"""
	#print("Total Time",time.time() - totTime)
	return ((bigMap-minMap), np.uint8(255*(bigMap-minMap)/(maxMap-minMap)))

def allAbs():
	for i in range(14,16):
		img = str(i)+".png"
		image = cv2.imread(img)
		for j in range(100,401,100):
			abst = abstract(image,j)
			cv2.imwrite("abs/"+str(j)+"/"+str(i)+img, abst)

def blurTest():
	image = cv2.imread("lena.bmp");
	channels = []
	for c in cv2.split(image):
		channels.append(GBlur(c,255,30))
	result2 = cv2.merge(channels)
	cv2.imwrite("lena-blur.jpeg", result2);

def distTest():
	image = cv2.imread("1-abs-1.png");
	result2 = distribution2(image,20,1);
	cv2.imwrite("distTest.png", result2);


#cv2.imwrite("redblack-map.jpeg", analyzeImg("redblack.jpeg"));
#cv2.imwrite("abc-map.jpeg", analyzeImg("abc.jpg"));
#cv2.imwrite("lena-map.jpeg", analyzeImg("lena.bmp"));
#cv2.imwrite("sparks-map.jpeg", analyzeImg("sparks.jpg"));
#cv2.imwrite("black_light-map.jpeg", analyzeImg("black_light.jpg"));

def stdTest(scale = 2):
	for i in range(1,14):
		img = str(i)
		image0 = cv2.imread(img+".png")
		image = cv2.imread("abs/400/"+img+img+".png")
		result1 = rarity(image,0.25)
		for j in range (5,140,5):
			result2 = distribution2(image, j, scale)
			result = remap(result1,result2[0]/(len(result2[0])*len(result2[0][0])),6)
			result = anotherMap(image0,result[0])
			cv2.imwrite("stdScale"+str(scale)+"/"+str(i)+"-std"+str(j)+"-"+img+".png", result)

def kTest(scale = 2,std = 40):
	for i in range(1,14):
		img = "abs-"+str(i)+".png"
		image = cv2.imread(img)
		result1 = rarity(image)
		result2 = distribution(image, std, scale)
		for j in range (0,60):
			result = remap(result1,result2,j/200.0)
			cv2.imwrite("kScale"+str(scale)+"/"+str(i)+"-std"+str(std)+"-k"+str.format('{0:.3f}', j/200.0)+"-dmap+umap+"+img, result)

def coKandSTD(scale = 10):
	for i in range(1,14):
		img = "abs-"+str(i)+".png"
		image = cv2.imread(img)
		result1 = rarity(image)
		for j in range (1,101,5):
			result2 = distribution2(image, j, scale)
			for k in range (0,8):
				result = remap(result1,result2,k/200.0)
				cv2.imwrite("coKandSTDScale"+str(scale)+"/"+str(i)+"/k"+str.format('{0:.3f}', k/200.0)+"-std"+str(j)+"-dmap+umap+"+img, result)

def anotherMap(img0,img1):
	a = len(img0)
	b = len(img0[0])
	r = np.zeros((a,b), np.double)
	print(a,b)
	maxR = 0.0
	minR = 255
	small_image = img0#cv2.cvtColor(np.float32(img0/255.0),cv2.COLOR_BGR2LAB)
	S_ = np.zeros((256,256,256), np.double)
	W = np.zeros((256,256,256), np.double)
	std = 6
	#S = nd.filters.gaussian_filter(img1, std, mode='constant', cval=0.0)
	S = cv2.GaussianBlur(img1, (b-1+b%2,a-1+a%2), std, borderType=cv2.BORDER_REPLICATE)
	for i in range(a):
		for j in range(b):
			l_ = int(small_image[i][j][0])
			a_ = int(small_image[i][j][1])
			b_ = int(small_image[i][j][2])
			S_[l_][a_][b_] += S[i][j]
			W[l_][a_][b_] += 1
	S_ = nd.filters.gaussian_filter(S_, std, mode='constant', cval=0.0)
	W = nd.filters.gaussian_filter(W, std, mode='constant', cval=0.0)
	for i in range(a):
		for j in range(b):
			l_ = int(small_image[i][j][0])
			a_ = int(small_image[i][j][1])
			b_ = int(small_image[i][j][2])
			r_ = S_[l_][a_][b_]/W[l_][a_][b_]
			r[i,j] = r_
			if(maxR < r_):
				maxR = r_
			if(minR > r_):
				minR = r_
	return np.uint8(255*((r-minR)/(maxR-minR)))

def remap(r1,r2,k=0.02):
	a = len(r1)
	b = len(r1[0])
	r = np.zeros((a,b), np.double)
	#print(a,b)
	maxR = 0.0
	minR = 255
	for i in range(a):
		for j in range(b):
			r_ = r1[i,j]/1.0*np.exp(-k*(r2[i,j]/1.0))
			r[i,j] = r_
			if(maxR < r_):
				maxR = r_
			if((minR > r_) and (minR > 0)):
				minR = r_
	return ((r-minR),np.uint8(255*((r-minR)/(maxR-minR))))

def analyzeImg(img):
	image0 = cv2.imread(img+".png")
	image = cv2.imread("abs/400/"+img+img+".png")
#	image = abstract(image0, 300)
#	cv2.imwrite(img+"-abs.png", image)
	cv2.imwrite("Umap-"+img+".png", rarity(image,0.25))
	result1 = rarity(image)
	result2 = distribution2(image, 17, 1)
	cv2.imwrite("Dmap-"+img+".png", ~result2[1])
	result = remap(result1,result2[0]/(len(result2[0])*len(result2[0][0])), 6)
	true_result = anotherMap(image0,result[0])
	cv2.imwrite("Teste-"+img+".png", true_result)
	return (result[1])

def doImages():
	start_time = time.time()
	for i in range(1,18):
		print("abs-image " + str(i)+".png")
		cv2.imwrite(str(i)+"-abs+map+dist-"+str(i)+".png", analyzeImg(str(i)));
		print("Time per image:", (time.time()-start_time)/i)

def testControl():
	cv2.imwrite("17"+"-abs+map+dist-"+"17"+".png", analyzeImg("17"));

#allAbs()
doImages()
#coKandSTD(1)
#stdTest(1)
#blurTest()

#testControl()

def profile():
	return cProfile.run("test.doImages()")