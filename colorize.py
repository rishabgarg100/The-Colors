import os
import sys
import cv2
import numpy as np
import tensorflow as tf

from skimage.io import imsave, imread
from skimage.transform import resize
from skimage import img_as_ubyte

def load_image(path):
	img = imread(path)
	# crop image from center
	short_edge = min(img.shape[:2])
	yy = int((img.shape[0] - short_edge) / 2)
	xx = int((img.shape[1] - short_edge) / 2)
	crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
	# resize to 224, 224
	img = resize(crop_img, (224, 224))
	# desaturate image
	return (img[:,:,0] + img[:,:,1] + img[:,:,2]) / 3.0
	# return img

def model():
	with open("model/colorize.tfmodel", mode='rb') as f:
		model = f.read()
	return model

def main():
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(model())
	grayscale = tf.placeholder(tf.float32, [1, 224, 224, 1])
	inferred_rgb, = tf.import_graph_def(graph_def,
		input_map={"grayscale": grayscale},
		return_elements=["inferred_rgb:0"])
	
	path=sys.argv[1]
	if any(x in path for x in ['.jpg','.TIF','.jpeg','.bmp']):
		with tf.Session() as sess:
			in_img = load_image(path)
			input_vector = in_img.reshape(1, 224, 224, 1)
			inferred_batch = sess.run(inferred_rgb, feed_dict={grayscale: input_vector})
			
			in_img = img_as_ubyte(in_img)
			out_img = img_as_ubyte(inferred_batch[0])
			imsave('./data/test/color/'+path, out_img)
	else:
		imgs = os.listdir(path)
		with tf.Session() as sess:
			for img in imgs:
				in_img = load_image(path+img)
				input_vector = in_img.reshape(1, 224, 224, 1)
				inferred_batch = sess.run(inferred_rgb, feed_dict={grayscale: input_vector})
				
				in_img = img_as_ubyte(in_img)
				out_img = img_as_ubyte(inferred_batch[0])
				imsave('./data/test/color/'+img, out_img)

if __name__ == '__main__':
	main()

