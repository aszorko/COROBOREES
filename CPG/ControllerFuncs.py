# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:18:16 2023

@author: alexansz
"""

import numpy as np
import math
from MathUtils import sig,hinge

class Controller:
    def __init__(self,body,cpg,brain,outw,outbias):
        self.body = body
        self.cpg = cpg
        self.brain = brain
        self.outw = outw
        self.outbias = outbias

    def getobs(self,obs):
        
        sidetilt = -obs[5]*obs[6] + obs[3]*obs[8]
        fronttilt = obs[7]
        tot_tilt = np.sqrt(sidetilt**2 + fronttilt**2)
    
        currheight = obs[1] - self.body.y0
        
        pardist = obs[2]
        perpdist = obs[0]
        
        orient = math.atan2(obs[6], obs[8]) 
        
        return pardist,perpdist,currheight,sidetilt,fronttilt,tot_tilt,orient
    
    def getactions(self,pos_diff,tilt_control,num_joints):
        #assumes three joints per limb. num_joints is total array size (padded with zeros)
        
        k = len(self.body.limblist)
        if self.body.dirflip: #tilt (-1 to 1) becomes sign of angles
            direction = tilt_control
            tilt_out = 0
        else:       #tilt added to leg angle
            direction = 1
            tilt_out = tilt_control                
        
        #tiltlimb=0 -> d=1; tiltlimb=1 -> d=direction
        d = [1 + self.body.tiltlimb[i]*(direction-1) for i in range(3)]
    
        leg_angs = np.array([sum(self.body.leg_amp*x*d[0]) for x in pos_diff])
        knee_angs = np.array([sum(self.body.knee_amp*x*d[1]) for x in pos_diff])
        hip_angs = np.array([sum(self.body.hip_amp*x*d[2]) for x in pos_diff])
    
        action = np.array([0.0 for i in range(num_joints)])
        
        action[:k] = self.body.limbdir[0]*(2*self.body.anglimit[0]*sig(2*leg_angs/self.body.anglimit[0]) + self.body.leg_zero + tilt_out*self.body.tiltlimb[0])
        action[k:2*k] = self.body.limbdir[1]*(2*self.body.anglimit[1]*sig(2*knee_angs/self.body.anglimit[1]) + self.body.knee_zero + tilt_out*self.body.tiltlimb[1])
        action[2*k:3*k] = self.body.limbdir[2]*(2*self.body.anglimit[2]*sig(2*hip_angs/self.body.anglimit[2]) + self.body.hip_zero + tilt_out*self.body.tiltlimb[2]) # + hip_fb
        
        return action
    
    
    def stepind(self,stepsperframe,dt,dtReal,dc,sidetilt,fronttilt,brain_filt):
        #dt = time step in cpg units
        #dtReal = time between frames (or updates) in seconds
        
        k = len(self.cpg.cons)
        j = len(self.cpg.cons[0].x)
        
        allbrainout = np.zeros([stepsperframe,k])
        allcpgout = np.zeros([stepsperframe,k,j])
        
        oldpos = np.array([hinge(c.x) for c in self.cpg.cons])
        
        for m in range(stepsperframe):
    
            A_fb = self.body.A_side_fb_amp*self.body.rightlimb*sidetilt + \
                self.body.A_front_fb_amp*self.body.frontlimb*(fronttilt)
            B_fb = self.body.B_side_fb_amp*self.body.rightlimb*sidetilt + \
                self.body.B_front_fb_amp*self.body.frontlimb*(fronttilt)
            
                
            if self.brain is not None:
                #filtered input goes via brain
                brainx = self.brain.step([brain_filt[m]],0,dt).squeeze()
                brain_out = hinge(brainx + self.outbias)
                cpg_in = np.matmul(self.outw,brain_out)
                allbrainout[m,:] = cpg_in
    
                z = [[A_fb[i], B_fb[i], cpg_in[i]] for i in range(k)]
            else:
                #filtered input goes directly (can still be zero)
                z = [[A_fb[i], B_fb[i], brain_filt[m]/10] for i in range(k)]
    
            out = self.cpg.step(z, dc, dt)
            allcpgout[m,:,:] = np.array([hinge(x) for x in out])
            
        newpos = allcpgout[-1,:,:]
        neur_diffs = (oldpos[self.body.limblist]-newpos[self.body.limblist])/dtReal
    
        return neur_diffs,allcpgout,allbrainout
    
