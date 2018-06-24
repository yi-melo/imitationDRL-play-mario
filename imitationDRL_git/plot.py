# -*- coding: utf-8 -*-
"""
Created on Tue May 15 15:42:16 2018

@author: yml
"""

import matplotlib.pyplot as plt
import numpy as np
import os

p1=np.load('/home/yml/tf/mario/imitationDRL/points/points518.npy')
p2=np.load('/home/yml/tf/mario/imitationDRL/points/points520.npy')
p3=np.load('/home/yml/tf/mario/imitationDRL/points/points515.npy')
p11=[]
p22=[]
p33=[]
for i in range(min(max([len(p1),len(p2),len(p3)]),800)):
    if(i+3<len(p1)-1):
        p11.append((p1[i]+p1[i+1]+p1[i+2]+p1[i+3])/4)
    if(i+4<len(p2)):
        p22.append((p2[i]+p2[i+1]+p2[i+2]+p2[i+3])/4)
    if(i+4<len(p3)):
        p33.append((p3[i]+p3[i+1]+p3[i+2]+p3[i+3])/4)
x1=np.arange(0,len(p11),1)
y1=p11
x2=np.arange(0,len(p22),1)
y2=p22
x3=np.arange(0,len(p33),1)
y3=p33

step=30
x11=x1[: : step]
y11=y1[: : step]
x22=x2[: : step]
y22=y2[: : step]
x33=x3[: : step]
y33=y3[: : step]

plt.text(10, 17, 'blue:0.6-0.1')
plt.text(10, 15, 'red:0.1')
plt.text(10, 13, 'green:1-0.1')
plt.ylabel('checkpoints')
plt.xlabel('experiments')
plt.plot(x11,y11,'b*-')
plt.plot(x22,y22,'rx-')
plt.plot(x33,y33,'gx-')
plt.show()
