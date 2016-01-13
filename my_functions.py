""" In this file are the functions implemented by me without all the optmization of scipy"""

import numpy as np
import cv2
import math
import operator
import array
import random
import time
import gc
from complementary_functions import *


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
	channels = []
	for c in cv2.split(img):
		channels.append(GBlur(c,min(a,b)/2 -1,25))
	blur_image = cv2.merge(channels)

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

	bigMap = 255*(rarityMapFake-minR)/(maxR-minR)
	return bigMap
