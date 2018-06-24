#!/usr/bin/env python
# -- coding: utf-8 --

import numpy as np
from termcolor import cprint
lap_point=[203,53]
check_ofset=[3,0]

class Reward:
    LAP_COLOR_MAP = {(214, 157, 222): 1,
                     (224, 162, 229): 2,
                     (65, 49, 66): 3}

    CHECKPOINTS = [
        #### Straight-away ####
        [(563, 317), (564, 317), (565, 317), (566, 317), (567, 317)],
        #### Around the first bend ####
        [(563, 285), (564, 285), (565, 285), (566, 285), (567, 285)],
        [(540, 259), (540, 260), (540, 261), (540, 262), (540, 263)],
        [(516, 286), (517, 286), (518, 286), (519, 286), (520, 286)],
        ##### The angled sections on etiher side of the tunnel #####
        ##add
        [  (534,308), (535,308), (535,307), (536,307), (536,306) ],
        ##
        [(553, 321), (554, 321), (554, 320), (555, 320), (555, 319)],
        ##add
        [ (558, 351), (559, 351), (560, 351), (561, 351)],
        ##
        [(547, 370), (548, 371), (549, 372), (550, 373)],
        #### They're making another left turn ####
        [(526, 397), (527, 397), (528, 397), (529, 397), (530, 397)],
        [(546, 412), (546, 413), (546, 414), (546, 415), (546, 416)],
        [(562, 398), (563, 398), (564, 398), (565, 398), (566, 398)],
        #### Straight-away to the line ####
        [(563, 380), (564, 380), (565, 380), (566, 380), (567, 380)]
        ]

    DEFAULT_STEP_REWARD = -1
    LAP_REWARD = 100
    CHECKPOINT_REWARD = 100
    END_REWARD = 1000
    END_DETECTION_REWARD_REFUND = 215
    
    BUMP_REWARD=-20
    
    END_EPISODE_THRESHOLD = 30
    
    def __init__(self):
        self.numpy_array=None
        self.lap=1
        self.end_episode_confidence=0
        self._checkpoint_tracker = [[False for i in range(len(self.CHECKPOINTS))] for j in range(3)]
    
    def _get_reward(self):
        cur_lap = self._get_lap()
    
        bump=self._get_bump()        
        
        cur_ckpt = self._get_current_checkpoint()
        
        self.episode_over=self._evaluate_end_state()
        #cprint('Get Reward called!','yellow')
        if self.episode_over:
            # Refund the reward lost in the frames between the race finish and end episode detection
            return self.END_DETECTION_REWARD_REFUND + self.END_REWARD
        else:
            if cur_lap != self.lap:
                self.lap = cur_lap
                cprint('Lap %s!' % self.lap, 'red')
                return self.LAP_REWARD
  
            if bump:return self.BUMP_REWARD    
            elif cur_ckpt > -1 and not self._checkpoint_tracker[self.lap - 1][cur_ckpt]:

                # Only allow sequential forward achievement, no backward or skipping allowed. 
                # e.g. If you hit checkpoint 6, you must have hit all prior checkpoints (1-5)
                #      on this lap for it to count
                if not all(self._checkpoint_tracker[self.lap - 1][:-(len(self.CHECKPOINTS)-cur_ckpt)]):
                    #cprint('CHECKPOINT hit but not achieved (not all prior points were hit)!', 'red')
                    return self.DEFAULT_STEP_REWARD

                cprint('CHECKPOINT achieved!', 'red')
                self._checkpoint_tracker[self.lap - 1][cur_ckpt] = True
                return self.CHECKPOINT_REWARD
            else:
                return self.DEFAULT_STEP_REWARD
                
    def _getPixelColor(self, image_array, x, y):
        base_pixel = image_array[y][x]
        red = base_pixel[0]
        green = base_pixel[1]
        blue = base_pixel[2]
        return (red, green, blue)    
        
    def _get_lap(self):
        pix_arr = self.numpy_array
        point_a = self._getPixelColor(pix_arr, lap_point[0], lap_point[1])
        if point_a in self.LAP_COLOR_MAP:
            return self.LAP_COLOR_MAP[point_a]
        else:
            # TODO: What should this do? The pixel is not known, so assume same lap?
            return self.lap
    
    def _get_current_checkpoint(self):
        cps = map(self._checkpoint, self.CHECKPOINTS)
        if any(cps):
            #cprint('--------------------------------------------','red')
            #cprint('Checkpoints: %s' % cps, 'yellow')

            checkpoint = np.argmax(cps)

            #cprint('Checkpoint: %s' % checkpoint, 'cyan')

            return checkpoint
        else:
            # We're not at a checkpoint
            return -1

    def _checkpoint(self, checkpoint_points):
        pix_arr = self.numpy_array
        colored_dots = map(lambda point: self._getPixelColor(pix_arr, point[0]+check_ofset[0], point[1]+check_ofset[1]), 
                           checkpoint_points)
        pixel_means = np.mean(colored_dots, 1)
        #print colored_dots
        #cprint('Pixel means: %s' % pixel_means, 'cyan')
        return any(val < 120 for val in pixel_means)

    def _evaluate_end_state(self):
        #cprint('Evaluate End State called!','yellow')
        pix_arr = self.numpy_array

        upper_left = self._getPixelColor(pix_arr, 19+check_ofset[0], 19+check_ofset[1])
        upper_right = self._getPixelColor(pix_arr, 620+check_ofset[0], 19+check_ofset[1])
        bottom_left = self._getPixelColor(pix_arr, 19+check_ofset[0], 460+check_ofset[1])
        bottom_right = self._getPixelColor(pix_arr, 620+check_ofset[0], 460+check_ofset[1])

        if upper_left == upper_right == bottom_left == bottom_right:
            self.end_episode_confidence += 1
        else:
            self.end_episode_confidence = 0

        if self.end_episode_confidence > self.END_EPISODE_THRESHOLD:
            return True
        else:
            return False
            
    def _get_bump(self):
        pix_arr = self.numpy_array
        pix = self._getPixelColor(pix_arr, 320+check_ofset[0], 240+check_ofset[1])
        
        if pix[0]>pix[1] and pix[0]>pix[2]:
            pix1= self._getPixelColor(pix_arr,275+check_ofset[0],330+check_ofset[1])#310
            pix2= self._getPixelColor(pix_arr,365+check_ofset[0],330+check_ofset[1])
        else:
            pix1= self._getPixelColor(pix_arr,275+check_ofset[0],310+check_ofset[1])#310
            pix2= self._getPixelColor(pix_arr,365+check_ofset[0],310+check_ofset[1])         
            
        temp=sorted(list(pix))
        delt=temp[2]-temp[0]
        temp1=sorted(list(pix1))
        delt1=temp1[2]-temp1[0]
        temp2=sorted(list(pix2))
        delt2=temp2[2]-temp2[0]
        
        bump_count=0
        if delt>50 and temp[2]>100 or 100<temp[0]:
            bump_count+=1;
        if delt1>50 and temp1[2]>100 or 100<temp1[0]:
            bump_count+=1;
        if delt2>50 and temp2[2]>100 or 100<temp2[0]:
            bump_count+=1;
        if bump_count>=3:
            return 1
        else: return 0
