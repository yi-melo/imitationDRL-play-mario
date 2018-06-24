#!/usr/bin/env python
# -- coding: utf-8 --

import numpy as np
import os
from skimage.io import imread
from reward import Reward
#samples dir
file_dir=os.listdir(os.getcwd()+'/samples')

get_r=Reward()

#obs,a,r,end
expert=[[],[],[],[]]
'''
expert=list(np.load('expert.npy'))
expert[0]=list(expert[0])
expert[1]=list(expert[1])
expert[2]=list(expert[2])
expert[3]=list(expert[3])
'''
r=[]
def get_reward(img):
    get_r.numpy_array=img
    r=get_r._get_reward()
    return r
#get every obs's a,r,end
#1' get obs and action from data.csv
# X for obs ,Y for action
def load_sample(sample):
    image_files = np.loadtxt(sample + '/data.csv', delimiter=',', dtype=str, usecols=(0,))
    joystick_values = np.loadtxt(sample + '/data.csv', delimiter=',', usecols=(1,2,3))
    return image_files, joystick_values

if __name__=='__main__':
  for i in range(len(file_dir)):
      # for sample in samples:
      img_file,joy_value=load_sample(os.getcwd()+'/samples/'+file_dir[i])
      j=0
      for img in img_file:
          #obs
          obs=imread(img)

          #action
          if joy_value[j][0]<0.003:
              action=[-80,0,1,0,0]
          else:
              if joy_value[j][0]>0.006:
                  action=[80,0,1,0,0]
              else:
                  action=[0,0,1,0,0]

          reward=get_reward(obs)
          r.append(reward)
          j+=1

          expert[0].append(obs)
          expert[1].append(action)
          expert[2].append(reward)
          expert[3].append(0)

      expert[3][len(expert[3])-1]=1

  np.save('expert.npy',expert)

