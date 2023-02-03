import cv2
import numpy as np
from matplotlib import pyplot as plt




# loading images
img0 = cv2.imread('PET-scan.tif', 0) #以灰階方式讀入圖片

# remove noises
img = cv2.GaussianBlur(img0,(3,3),0) #高斯模湖水怪

# convolution with proper kernel
laplacian = cv2.Laplacian(img,cv2.CV_64F) #(b)的答案
#(CV_64F = 64-bit ﬂoating-point numbers )
imgadd = cv2.add(img,laplacian,dtype=cv2.CV_64F) #(c) 的答案
# (CV_64F = 64-bit ﬂoating-point numbers)
x = cv2.Sobel(img, cv2.CV_16S, 1, 0 , ksize=5)   
# 對x求一階導 #(d) 的答案
# y (ksize = 5  --> 5x5ㄉfilter)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1 , ksize=5)   
# 對y求一階導 #(d) 的答案
# y (ksize = 5  --> 5x5ㄉfilter)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0) #(d) 的答案

blur = cv2.blur(Sobel, (5, 5) ) #(e) 的答案

mask =cv2.multiply (laplacian,blur ,dtype=cv2.CV_64F) #(f) 的答案


mask = cv2.convertScaleAbs(mask)

imgadd2 = cv2.add(img,mask,dtype=cv2.CV_64F)






#一次show全部
# plt.subplot(1,6,1),plt.imshow(img,cmap = 'gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])

# plt.subplot(1,6,2),plt.imshow(laplacian,cmap = 'gray')
# plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

# plt.subplot(1,6,3),plt.imshow(imgadd,cmap = 'gray')
# plt.title('imgadd'), plt.xticks([]), plt.yticks([])



plt.show()

#分開秀，要一張一張關
plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(imgadd,cmap = 'gray')
plt.title('imgadd'), plt.xticks([]), plt.yticks([])
plt.show()


plt.imshow(Sobel,cmap = 'gray')
plt.title('Sobel'), plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(blur,cmap = 'gray')
plt.title('blur'), plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(mask,cmap = 'gray')
plt.title('mask'), plt.xticks([]), plt.yticks([])
plt.show()


plt.imshow(imgadd2,cmap = 'gray')
plt.title('imgadd2'), plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(imgadd2,cmap = 'gray')
plt.title('imgadd2'), plt.xticks([]), plt.yticks([])
plt.show()


for gamma in [0.1,0.5,1.2,2.2]:
    gamma_corrected = np.array(255*(imgadd2/255)**gamma,dtype='uint8')
    plt.imshow(gamma_corrected,cmap = 'gray')
    plt.title('gamma_corrected'+str(gamma)), plt.xticks([]), plt.yticks([])
    plt.show()
