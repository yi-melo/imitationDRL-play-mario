#!/usr/bin/env python

from utils import resize_image, XboxController
from termcolor import cprint
import gym, gym_mupen64plus
from train_expert import create_model
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
from scipy import misc

def GetPixelColor(image_array, x, y):
        base_pixel = image_array[y][x]
        red = base_pixel[0]
        green = base_pixel[1]
        blue = base_pixel[2]
        return (red, green, blue)

# Play
class Actor(object):

    def __init__(self):
        # Load in model from train.py and load in the trained weights
        self.model = create_model(keep_prob=1) # no dropout
        self.model.load_weights('model_weights.h5')

        # Init contoller for manual override
        self.real_controller = XboxController()

    def get_action(self, obs):

        ### determine manual override
        manual_override = self.real_controller.LeftBumper == 1
        go=self.real_controller.RightBumper==1
        
        if not manual_override:
            ## Look
            vec = resize_image(obs)
            vec = np.expand_dims(vec, axis=0) # expand dimensions for predict, it wants (1,66,200,3) not (66, 200, 3)
            ## Think
            joystick = self.model.predict(vec, batch_size=1)[0]

        else:
            joystick = self.real_controller.read()
            joystick[1] *= -1 # flip y (this is in the config when it runs normally)


        ## Act

        ### calibration
        if manual_override:
            output = [
                (int(joystick[0] * 10000)-40)*2,
                int(joystick[1] * 10000)-40,
                int(joystick[2]),
                int(round(joystick[3])),
                int(round(joystick[4])),
                ]
        else:
            output = [
                round(joystick[0])*40,
                0,
                int(joystick[2]+1),
                0,
                0,
                ]
        ### print to console
        if manual_override:
            cprint("Manual: " + str(output), 'yellow')
        else:
            cprint("AI: " + str(output), 'green')
        
        if not go and not manual_override:
            output[2]=0
        return output


if __name__ == '__main__':
    env = gym.make('Mario-Kart-Luigi-Raceway-v0')

    obs = env.reset()
    env.render()
    print('env ready!')

    actor = Actor()
    print('actor ready!')

    print('beginning episode loop')
    total_reward = 0
    end_episode = False
    while not end_episode:
        action = actor.get_action(obs)
        obs, reward, end_episode, info = env.step(action)
        env.render()
        total_reward += reward
        im1 = misc.imresize(obs, (480,640))
        pix=GetPixelColor(im1,320,240)
        print pix
        temp=sorted(list(pix))
        delt=temp[2]-temp[0]
        if delt>50 and temp[2]>100:
            bump=1
        else: bump=0
        print bump
    print('end episode... total reward: ' + str(total_reward))

    obs = env.reset()
    print('env ready!')

    input('press <ENTER> to quit')

    env.close()
