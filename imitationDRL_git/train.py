# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 09:52:35 2018

@author: yml
"""

#!/bin/python
import gym, gym_mupen64plus
import cv2
import numpy as np
import os
from model import Brain

import tensorflow as tf

import matplotlib.pyplot as plt
from utils import resize_image

filename = 'write_data.txt'

'''
expert=list(np.load('expert.npy'))
expert[0]=list(expert[0])
expert[1]=list(expert[1])
expert[2]=list(expert[2])
expert[3]=list(expert[3])
'''


# preprocess raw image to 80*80 gray image
def preprocess(observation):
	observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
	#ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
	return np.reshape(observation,(80,80,1))

def playGame():
    env = gym.make('Mario-Kart-Luigi-Raceway-v0')
    env.reset()
    env.render()

    actions = 3
    brain = Brain(actions)
    for i in range(20):
        (obs, rew, end, info) = env.step([0, 0, 0, 0, 0]) # NOOP until green light
        env.render()

    vec = resize_image(obs)
    brain.currentObs=vec

    obs = cv2.cvtColor(cv2.resize(obs, (80, 80)), cv2.COLOR_BGR2GRAY)
    brain.setInitState(obs)
    rew_sum=0


    test=0  #if test learning result
    point=0 #checkpoints if can get

    pointArr=[]
    #pointArr=np.load('points.npy')
    #pointArr=list(pointArr)
    while 1!= 0:
        if  brain.timeStep>0 and brain.timeStep%2000==0:
            test=4
        test=1
        action = brain.getAction(test)
        #actual action
        if (action==[1,0,0]).all():
            (obs, rew, end, info) = env.step([0,   0, 1, 0, 0]) # Drive straight
        if (action==[0,1,0]).all():
            (obs, rew, end, info) = env.step([-80, 0, 1, 0, 0])
        if (action==[0,0,1]).all():
            (obs, rew, end, info) = env.step([80, 0, 1, 0, 0])

        if rew>0:
            rew_sum=0
            if rew==100:
                point+=1
                if point==36:
                    end=1
        else:
            rew_sum+=rew

        vec = resize_image(obs)
        brain.currentObs=vec
        obs = preprocess(obs)

        if rew_sum<-400:
            end=1
            rew=-10
            rew_sum=0

        if end==1:
            env.reset()
            if test>0:
                pointArr.append(point)
                np.save('points',pointArr)
                test-=1
            point=0
        rew+=abs(brain.flagSame*4)*brain.alpha

        print rew
        brain.setPerception(obs,action,rew,end)

        env.render()
    env.close()

def main():
	playGame()

if __name__ == '__main__':
	main()

