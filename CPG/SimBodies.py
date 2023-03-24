# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:25:17 2023

@author: alexansz
"""

import numpy as np

class QuadBody:
    def __init__(self,bodytype,pint):

        p = [float(x)/10 for x in pint]

        if bodytype=='shortquad':
            self.ymax = 0.2
        else:
            self.ymax = 0.4
            
        self.y0 =-0.48
    
        self.anglimit = [1,1,1] #amplitude limit in radians    
    
        self.hip_zero = -0.3*p.pop()  #zero-amplitude angle in radians
        self.leg_zero = 0.5*p.pop()   
        self.knee_zero = -0.8*p.pop()   
    
        self.leg_amp = np.array([0.05*p.pop(), 0, 0])  
        self.knee_amp = np.array([0, 0.05*p.pop(), 0])
        self.hip_amp = np.array([0, 0, 0])
        
        self.B_front_fb_amp = -0.55 + p.pop()
        self.B_side_fb_amp = -0.55 + p.pop()
        self.A_front_fb_amp = -0.55 + p.pop()
        self.A_side_fb_amp = -0.55 + p.pop() 
    
        # matsuoka_quad limb order = LH,RH,LF,RF
        # Unity limb order = RF,LF,RH,LH
        self.limblist = [3, 2, 1, 0]  # reordering from CPG script
        #below arrays are used for feedbacks
        self.frontlimb = np.array([1, 1, -1, -1])
        self.rightlimb = np.array([1, -1, 1, -1])
        self.limbdir = np.array([-self.rightlimb,-self.rightlimb,[1,1,1,1]])  # leg+knee hinges are reversed between left-right in Unity
        self.tiltlimb = [1,0,0]
        
        self.dirflip = False
        self.audioamp = 1.0

        if len(p)>0:
            raise ValueError(f'p is the wrong length. {len(p)} values left')

class HexBody:
    def __init__(self,bodytype,pint):

        p = [float(x)/10 for x in pint]

        self.ymax = 0.1
        self.y0 = -0.65
    
        self.anglimit = [0.5,0.5,0.3] #amplitude limit in radians    
    
        self.hip_zero = 0
        self.leg_zero = (-0.55 + p.pop())*0.3 #0.5
        self.knee_zero = (-0.55 + p.pop())*0.3 #0.5
    
        self.leg_amp = np.array([0.1*(-0.55+p.pop()), 0, 0])  
        self.knee_amp = np.array([0.1*(-0.55+p.pop()), 0, 0])
        self.hip_amp = np.array([0, 0.1*p.pop(), 0])
        
        self.B_front_fb_amp = -0.55 + p.pop()
        self.B_side_fb_amp = -0.55 + p.pop()
        self.A_front_fb_amp = -0.55 + p.pop()
        self.A_side_fb_amp = -0.55 + p.pop() 
    
        #order = Left front/middle/hind, Right front/middle/hind
        self.limblist = [0,1,2,3,4,5]  # same order as CPG script
        #below arrays are used for feedbacks
        self.frontlimb = np.array([1, 0, -1, 1, 0, -1])
        self.rightlimb = np.array([-1, -1, -1, 1, 1, 1])
        self.limbdir = np.ones([3,6])
        self.tiltlimb = [0,0,1]

        self.dirflip = True
        self.audioamp = 0.5

        if len(p)>0:
            raise ValueError(f'p is the wrong length. {len(p)} values left')