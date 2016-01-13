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
import gc
from complementary_functions import *

def abstract(img,K=100, M_tol=0.5):
	""" K-means abstraction """
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


def rarity(img,number=0.25):
	""" Calculate the rarity/uniqueness of a given image """
	a = len(img)
	b = len(img[0])
	n_pixels = a*b
	start_time = time.time()
	
	small_image = cv2.cvtColor(np.float32(img/255.0),cv2.COLOR_BGR2LAB)
	small_image = small_image/1.0;

	w=b-1+(b%2)
	h=a-1+(a%2)
	blur_image2 = np.zeros((a,b), np.double)
	for i in range(a):
		for j in range(b):
			p = small_image[i,j]/1.0;
			blur_image2[i,j] = np.dot(p,p)
	blur_image = cv2.GaussianBlur(small_image, (w,h), number*math.sqrt(n_pixels), borderType=cv2.BORDER_REPLICATE)
	blur_image2 = cv2.GaussianBlur(blur_image2, (w,h), number*math.sqrt(n_pixels), borderType=cv2.BORDER_REPLICATE)
	
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
			rarityMapFake[i,j] = r
			if(r < minR):
				minR = r
			if(r > maxR):
				maxR = r
	
	bigMap = 255*(rarityMapFake-minR)/(maxR-minR)
	return bigMap


def distribution2(img,std = 20, scale = 2, kernel_size = 253):
	""" Calculate the spatial variance of color """
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
	#commands for scipy < 1.9:
	#m1[:,:,:,0] = nd.filters.gaussian_filter(midMapi, std, mode='wrap', cval=0.0)
	#m1[:,:,:,1] = nd.filters.gaussian_filter(midMapj, std, mode='wrap', cval=0.0)
	#m1_ = nd.filters.gaussian_filter(m2_, std, mode='wrap', cval=0.0)
	#better commands for scipy > 1.9:
	trunc = (len(midMapi)/std)
	m1[:,:,:,0] = nd.filters.gaussian_filter(midMapi, std, mode='constant', cval=0.0, truncate=trunc)
	m1[:,:,:,1] = nd.filters.gaussian_filter(midMapj, std, mode='constant', cval=0.0, truncate=trunc)
	m1_ = nd.filters.gaussian_filter(m2_, std, mode='constant', cval=0.0, truncate=trunc)
	w = nd.filters.gaussian_filter(w, std, mode='constant', cval=0.0, truncate=trunc)
	maxMap = 0
	minMap = 0
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
	return ((bigMap-minMap), np.uint8(255*(bigMap-minMap)/(maxMap-minMap)))

def anotherMap(img0,img1):
	""" This function maps the result of salience assignement to the final image """
	a = len(img0)
	b = len(img0[0])
	r = np.zeros((a,b), np.double)
	print(a,b)
	maxR = 0.0
	minR = 255
	small_image = img0
	S_ = np.zeros((256,256,256), np.double)
	W = np.zeros((256,256,256), np.double)
	std = 6
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
	""" This function maps the rarity/uniqueness and the distribution into a saliency assignement"""
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

def analyzeImg(img, rarity_dev = 0.25, dist_dev = 17, remap_k = 6, scale = 1):
	image0 = cv2.imread(img+".png")
	image = cv2.imread("abs/400/"+img+img+".png")
	#cv2.imwrite("Umap-"+img+".png", rarity(image,rarity_dev))
	result1 = rarity(image)
	result2 = distribution2(image, dist_dev, scale)
	#cv2.imwrite("Dmap-"+img+".png", ~result2[1])
	result = remap(result1,result2[0]/(len(result2[0])*len(result2[0][0])), remap_k)
	true_result = anotherMap(image0,result[0])
	#cv2.imwrite("Teste-"+img+".png", result[1])
	return (true_result)

def analyzeImgWithAbstraction(img, abstractK = 100, rarity_dev = 0.25, dist_dev = 17, remap_k = 6, scale = 1):
	image0 = cv2.imread(img+".png")
	image = abstract(image0, abstractK)
	#cv2.imwrite(img+"-abs.png", image)
	#cv2.imwrite("Umap-"+img+".png", rarity(image,rarity_dev))
	result1 = rarity(image)
	result2 = distribution2(image, dist_dev, scale)
	#cv2.imwrite("Dmap-"+img+".png", ~result2[1])
	result = remap(result1,result2[0]/(len(result2[0])*len(result2[0][0])), remap_k)
	true_result = anotherMap(image0,result[0])
	#cv2.imwrite("Teste-"+img+".png", result[1])
	return (true_result)
