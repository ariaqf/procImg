from salience import *

def blurTest():
	"""Test of Blur Function"""
	image = cv2.imread("lena.bmp");
	channels = []
	for c in cv2.split(image):
		channels.append(GBlur(c,255,30))
	result2 = cv2.merge(channels)
	cv2.imwrite("lena-blur.jpeg", result2);

def distTest():
	""" Test of Distribution algorithm """
	image = cv2.imread("1.png");
	result2 = distribution2(image,20,1);
	cv2.imwrite("distTest.png", result2);

def stdTest(scale = 2):
	""" Search for a good distribution standard deviation """
	for i in range(1,16):
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
	""" Search for a good k for the remap algorithm """
	for i in range(1,16):
		img = "abs-"+str(i)+".png"
		image = cv2.imread(img)
		result1 = rarity(image)
		result2 = distribution(image, std, scale)
		for j in range (0,60):
			result = remap(result1,result2,j/200.0)
			cv2.imwrite("kScale"+str(scale)+"/"+str(i)+"-std"+str(std)+"-k"+str.format('{0:.3f}', j/200.0)+"-dmap+umap+"+img, result)

def doImages():
	""" Make all test images (1-17) """
	start_time = time.time()
	for i in range(1,16):
		print("abs-image " + str(i)+".png")
		cv2.imwrite(str(i)+"-abs+map+dist-"+str(i)+".png", analyzeImg(str(i)));
		print("Time per image:", (time.time()-start_time)/i)

def testControl():
	cv2.imwrite("1"+"-abs+map+dist-"+"1"+".png", analyzeImg("1"));

def allAbs( k = 0):
	""" Make an abstract of every test image """
	for i in range(1,16):
		img = str(i)+".png"
		image = cv2.imread(img)
		if(k == 0):
			for j in range(100,701,100):
				abst = abstract(image,j)
				cv2.imwrite("abs/"+str(j)+"/"+str(i)+img, abst)
		else:
			abst = abstract(image,k)
			cv2.imwrite("abs/"+str(k)+"/"+str(i)+img, abst)
