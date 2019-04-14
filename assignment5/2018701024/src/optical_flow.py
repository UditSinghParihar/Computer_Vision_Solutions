from sys import argv, exit
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from numpy.linalg import inv

def display_image(img, gray=False):
	fig = plt.figure("Image").add_subplot(1,1,1)
	fig.imshow(img, cmap="gray")
	plt.axis("off")
	plt.show()

def display_vel(img, vel):
	fig = plt.figure("Velocities").add_subplot(1,1,1)
	fig.imshow(img, cmap="gray")
	plt.axis("off")

	U = vel[:,:,0]
	V = vel[:,:,1]
	(H,W) = U.shape[:2]
	X = []
	Y = []

	counter = 1
	for y in range(0, H):
		for x in range(0, W):
			if U[y,x] != 0 and V[y,x] != 0 and (counter%2 == 0):
				X.append(x)
				Y.append(y)
			counter += 1
	# print(np.unique(U),2*'\n',np.unique(V))
	
	Q = plt.quiver(X,Y,U,V,color='g')
	plt.quiverkey(Q,0.9,0.9,1,r'$1 \frac{m}{s}$', labelpos='E')
	plt.show()

def calculate_vel(x,y,t):
	x = x.reshape(9,1)
	y = y.reshape(9,1)
	t = t.reshape(9,1)
	
	A = np.hstack((x,y))
	try:	
		u = inv(A.T.dot(A)).dot(A.T).dot(t)
		u_n = np.linalg.norm(u)
		if u_n < 10:
			return 0, 0
		# print(u_n)
		u = u/u_n
		return u[0] ,u[1]
	
	except np.linalg.LinAlgError as err:
		if 'Singular matrix' in str(err):
			calculate_vel.count += 1
			return 0, 0
		

calculate_vel.count=0

def segmentation(img, vel):
	v_norm = vel**2
	v_norm = v_norm[:,:,0] + v_norm[:,:,1]
	v_norm = v_norm**(0.5)
	# v_norm = rescale_intensity(v_norm, out_range=(0,255)).astype("uint8")
	print(np.unique(v_norm), v_norm.dtype, 2*'\n', v_norm)
	display_image(v_norm)
	# (ret, thresh) = cv2.threshold(v_norm)

def main(file1, file2):
	i1 = io.imread(file1)
	i2 = io.imread(file2)
	display_image(i1)
	display_image(i2)

	img1 = cv2.GaussianBlur(i1,(5,5),0)
	img2 = cv2.GaussianBlur(i2,(5,5),0)

	kernel_x = (1.0/8.0)*np.array([[-1,0,1],
							[-2,0,2],
							[-1,0,1]], dtype='float32')
	kernel_y = (1.0/8.0)*np.array([[1,2,1],
							[0,0,0],
							[-1,-2,-1]])

	img1_x = cv2.filter2D(img1, -1, kernel_x)
	img1_y = cv2.filter2D(img1, -1, kernel_y)
	img2_x = cv2.filter2D(img2, -1, kernel_x)
	img2_y = cv2.filter2D(img2, -1, kernel_y)

	img_x = (img1_x+img2_x)
	img_y = (img1_y+img2_y)
	img_t = (img2-img1)

	(H, W) = img1.shape
	vel = np.zeros((H,W,2))

	for y in np.arange(1, H-1):
		for x in np.arange(1, W-1):
			roi_x = img_x[y-1:y+2, x-1:x+2]
			roi_y = img_y[y-1:y+2, x-1:x+2]
			roi_t = img_t[y-1:y+2, x-1:x+2]
			vel[y,x] = calculate_vel(roi_x, roi_y, roi_t)

	print('Singular pixels: %d' % calculate_vel.count)

	display_vel(i1, vel)

	segmentation(i1, vel)

if __name__ == '__main__':
	if len(argv) < 3:
		print("Usage: python %s img1.png img2.png" % argv[0])
		exit()

	main(argv[1], argv[2])