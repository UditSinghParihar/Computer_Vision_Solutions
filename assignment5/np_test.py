import numpy as np
from skimage import io
from sys import argv
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
import cv2

class Greeter(object):
	def __init__(self, name):
		self.name = name

	def greet(self, loud=False):
		if loud:
			print('Hello, %s!' % self.name.upper())
		else:
			print('Hello, %s' % self.name)
		

def display_image(rgb, gray=False):
	fig = plt.figure("Image").add_subplot(1,1,1)
	if gray:
		fig.imshow(rgb, cmap="gray")
	else:
		fig.imshow(rgb)
	plt.axis("off")
	plt.show()

def main():
	a = np.array(range(10)).reshape(5,2)
	b = (a>5)
	# print(a,2*'\n',b)
	# print(a[b])

	person = Greeter('manas')
	# person.greet()

	e = np.array(range(1,5)).reshape(2,2)
	# f = np.array(range(2,6)).reshape(2,2)
	# print(e,2*'\n',f,2*'\n', np.dot(e,f))

	g = np.zeros(e.shape)
	# print(g)

	h = np.array(range(1,10)).reshape(3, 3)
	i = np.array([1,0,1])
	ii = np.tile(i, (3,1))
	# print(h + i)

	j = np.array(range(12)).reshape(2,3,2)
	k = np.array([1,2,1]).reshape(3,1)
	# print(j,2*'\n',k,2*'\n',j*k)

	l = np.arange(30).reshape(2,3,5)
	m = np.array([[True,True,False],[False,True,True]])
	# print(l.shape, 2*'\n', m.shape)

	c = np.array([[0,1],[1,0],[2,0]])
	d = squareform(pdist(c, 'euclidean'))
	# print(d)

def convolve(img, kernel):

	(iH, iW) = img.shape[:2]
	(kH, kW) = kernel.shape[:2]
	pad = (kH-1)//2

	img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
	output = np.zeros((iH, iW), dtype='float32')

	for y in np.arange(pad, pad+iH):
		for x in np.arange(pad, pad+iW):
			roi = img[y-pad:y+pad+1, x-pad:x+pad+1]
			k = (roi*kernel).sum()
			output[y-pad, x-pad] = k

	output = rescale_intensity(output, out_range=(0,255)).astype("uint8")
	# print(output.shape,2*'\n',np.unique(output))
	return output

def img_operation(file):
	img = io.imread(file)
	display_image(img)
	# print(np.unique(img),2*'\n',img.shape, img.dtype)

	tint = np.array([2,1,1])
	img = img*tint
	threshold = (img>255)
	img[threshold] = 255
	display_image(img)

	sobelX = np.array([[-1,0,1],
						[-2,0,2],
						[-1,0,1]], dtype='int')
	laplacian = np.array([[0,1,0],
							[1,-4,1],
							[0,1,0]], dtype='int')

	gray = rgb2gray(img)
	gray = rescale_intensity(gray, out_range=(0,1))
	display_image(gray, True)

	img1 = convolve(gray, sobelX)
	display_image(img1, True)

	img2 = convolve(gray, laplacian)
	display_image(img2, True)

	img3 = cv2.filter2D(gray, -1, laplacian)
	display_image(img3, True)

if __name__ == '__main__':
	main()
	img_operation(argv[1])