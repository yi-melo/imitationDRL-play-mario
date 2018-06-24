#!/usr/bin/env python

import array
import numpy as np
from PIL import Image
from skimage.util import img_as_float


def prepare_image(img):

    img = img.reshape(Screenshot.SRC_H, Screenshot.SRC_W, Screenshot.SRC_D)

    return resize_image(img)


def resize_image(img):

    im = Image.fromarray(img)
    im = im.resize((Screenshot.IMG_W, Screenshot.IMG_H))

    im_arr = np.frombuffer(im.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((Screenshot.IMG_H, Screenshot.IMG_W, Screenshot.IMG_D))

    return img_as_float(im_arr)


class Screenshot(object):
    SRC_W = 640
    SRC_H = 480
    SRC_D = 3

    OFFSET_X = 100
    OFFSET_Y = 100

    IMG_W = 200
    IMG_H = 66
    IMG_D = 3

    image_array = array.array('B', [0] * (SRC_W * SRC_H * SRC_D));


def load_sample(sample):
    image_files = np.loadtxt(sample + '/data.csv', delimiter=',', dtype=str, usecols=(0,))
    joystick_values = np.loadtxt(sample + '/data.csv', delimiter=',', usecols=(1,2,3,4,5))
    return image_files, joystick_values


if __name__=='__main__':
  print("Preparing data")

  X = []
  y = []

  data=list(np.load('expert.npy'))
  data[0]=list(data[0])
  data[1]=list(data[1])
  data[2]=list(data[2])
  data[3]=list(data[3])

  X=data[0]
  y=data[1]


  for i in range(len(X)):
      X[i]=prepare_image(X[i])
      if y[i][0]==-80:
          y[i]=np.asarray([0,1,0])
      elif y[i][0]==80:
          y[i]=[0,0,1]
      else:
          y[i]=[1,0,0]
  np.save("data/X", X)
  np.save("data/y", y)
  print("Done!")
