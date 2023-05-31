# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 11:16:46 2023

@author: alexansz
"""

from scipy import signal
import UnityInterfaceBrain
import MathUtils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from mlagents_envs.base_env import (
    ActionTuple
)

def wfunc(t,k=1,decay=0.1):
    return t*k*np.exp(-abs(t)/decay)

def main(env,controller,bodytype,z_in,period_in,nframes,dc,tilt,seed=123,plot=False):

       
    m = len(controller.cpg.cons)
    
    dt_real,dt,t0 = UnityInterfaceBrain.gettimestep(bodytype,True)
    stepsperframe = round(dt_real / dt / t0)
    period_frames = 2*round(period_in/dt_real)


    nframes_timelearn = nframes
    nframes_amplearn = nframes
    nframes_selfsync = nframes   
    corrstart = 0.5
    
    z = MathUtils.lpf(z_in,controller.decay*dt*stepsperframe)
    zmean = np.mean(z)

    
    #no input
    allbrainout,allintn,allcpgout,allsensors,zout,free_height,_,_ = run(env,controller,nframes,dc,tilt)

    zsens = MathUtils.lpf(np.sum(1+allsensors,axis=0),controller.decay*dt*stepsperframe)
    free_corr = crosscorr2(zsens,z,period_frames,start=corrstart)
    

    free_period,free_autocorr = MathUtils.autocorr(allcpgout, corrstart, mindelay=round(0.1/dt_real))


    controller.cpg.reset(seed)
    controller.brain.reset(seed+1)
               

    allbrainout,allintn,allcpgout,allsensors,zout,sync_height,_,_ = run(env,controller,nframes_timelearn,dc,tilt,brain_in=z[:nframes_timelearn])

    zsens2 = MathUtils.lpf(np.sum(1+allsensors,axis=0),controller.decay*dt*stepsperframe)
    sync_corr = crosscorr2(zsens2,z,period_frames,start=corrstart)
    syncfree_corr = crosscorr2(zsens2,zsens,period_frames,start=corrstart)

    fbamp = 0.0 #for initial amplitudes  
    
    sync_period,sync_autocorr = MathUtils.autocorr(allcpgout, corrstart, mindelay=round(0.1/dt_real))


    delays,corrheights,n_thresh,_ = analysis(allbrainout,allintn,allsensors,z[:nframes_timelearn],sync_period,m,dt_real,nframes_timelearn,stepsperframe,plot=plot)  
    
    #no good peaks -> end evaluation here
    if delays is None:
       if plot:
           print("No good peaks. End evaluation")
       return (zmean,free_height,dt_real*free_period,free_autocorr,free_corr,sync_height,dt_real*sync_period,sync_autocorr,sync_corr,syncfree_corr,-1,-1,-1,-1,-1,-1,-1,-1)

    controller.cpg.reset(seed)
    controller.brain.reset(seed+1)

    if plot:
       print("Learning amplitudes:")    

    
    allbrainout,allintn,allcpgout,allsensors,zout,_,amps_out,learnout = run(env,controller,nframes_amplearn,dc,tilt,brain_in=z[:nframes_amplearn],footfb = (delays,np.array(corrheights)*fbamp,1.0,n_thresh,zmean))


    
    if plot:
        print(amps_out)
        wind = 5
        startframe = round(nframes_amplearn*0.8)
        z_ma = np.convolve(z[startframe:nframes_amplearn],np.ones(wind)/wind,'same')
        fb_ma = np.convolve(zout[startframe*stepsperframe::stepsperframe],np.ones(wind)/wind,'same')
        fig, axs = plt.subplots(3,1)
        axs[0].plot(fb_ma)
        axs[0].plot(z_ma)
        axs[1].plot(z_ma-fb_ma)
        axs[2].plot(np.array([x[round(nframes_amplearn*0.8):]+i*1.5 for i,x in enumerate(learnout[2])]).T)
        plt.show()
        plt.figure()
        plt.plot(learnout[0].T)
        plt.show()

        print("Running in self-sync mode:")
    
    
    controller.cpg.reset(seed)
    controller.brain.reset(seed+1)
    
    allbrainout,allintn,allcpgout,allsensors,zout,learn_height,_,learnout = run(env,controller,nframes_selfsync,dc,tilt,footfb = (delays,amps_out,1.0,n_thresh,zmean))


    zsens3 = MathUtils.lpf(np.sum(1+allsensors,axis=0),controller.decay*dt*stepsperframe)
    learn_corr = crosscorr2(zsens3,z,period_frames,start=corrstart)    
    learn_period,learn_autocorr = MathUtils.autocorr(allcpgout, corrstart, mindelay=round(0.1/dt_real))
    
    learnfree_corr = crosscorr2(zsens3,zsens,period_frames,start=corrstart)
    learnsync_corr = crosscorr2(zsens3,zsens2,period_frames,start=corrstart)
    fb_corr = crosscorr2(zout[::stepsperframe],z,period_frames,start=corrstart)


    if plot:
        print("Mean feedback:",np.mean(zout))
        plt.figure()
        plt.plot(learnout[3])
        plt.show()
        wind = 5
        startframe = round(nframes_selfsync*0.8)
        fb_ma = np.convolve(zout[startframe*stepsperframe::stepsperframe],np.ones(wind)/wind,'same')
        fig, axs = plt.subplots(2,1)
        axs[0].plot(fb_ma)
        #axs[1].plot(np.array([x+i*1.5 for i,x in enumerate(learnout[1])]).T)
        axs[1].plot(np.array([x[round(nframes_selfsync*0.8):]+i*1.5 for i,x in enumerate(learnout[2])]).T)
        plt.show()
        analysis(allbrainout,allintn,allsensors,zout[:stepsperframe*nframes_selfsync:stepsperframe],learn_period,m,dt_real,nframes_selfsync,stepsperframe,plot=True)
        plotcorrs([z,zsens,zsens2,zsens3,zout[::stepsperframe]],period_frames,dt_real,['$z_T$','$x_F$','$x_S$','$x_L$','$z_L$'],'./allautocorrs.eps',start=corrstart)

    
    return (zmean,free_height,dt_real*free_period,free_autocorr,free_corr,sync_height,dt_real*sync_period,sync_autocorr,sync_corr,syncfree_corr,learn_height,dt_real*learn_period,learn_autocorr,learn_corr,learnfree_corr,learnsync_corr,fb_corr,np.mean(zout))


def analysis(allbrainout,allintn,allsensors,z,cpgperiod,m,dt_real,nframes,stepsperframe,plot=False):
    

    
    if plot:
        fig, axs = plt.subplots(4,1)
        
        tstart = round(0.8*nframes)
        tend = nframes
        
        axs[0].plot(z[tstart:tend])
        axs[1].plot(allbrainout[:,tstart:tend].T)
        axs[2].plot(allintn[:,tstart:tend].T)
        for k in range(m):
           axs[3].plot(allsensors[k,tstart:tend]+0.1*k)
        
        plt.show()
    
    
    
    inputpeak, _ = MathUtils.autocorr(np.array([z]), 0.33, mindelay=round(0.1/dt_real))
    
    filterpeak, _ = MathUtils.autocorr(allbrainout, 0.33, mindelay=round(0.1/dt_real))
    
    interpeak, _ = MathUtils.autocorr(allintn, 0.33, mindelay=round(0.1/dt_real))

    sensorpeak, _ = MathUtils.autocorr(allsensors, 0.33, mindelay=round(0.1/dt_real))

    outperiod = cpgperiod

    if plot:
        print(f"Input T = {inputpeak*dt_real:1.3f}") 
        print(f"Filter out T = {filterpeak*dt_real:1.3f}") 
        print(f"Interneurons T = {interpeak*dt_real:1.3f}") 
        print(f"CPG T = {cpgperiod*dt_real:1.3f}")
        print(f"Sensors T = {sensorpeak*dt_real:1.3f}") 
        print(f"Base T = {outperiod*dt_real:1.3f}")
        print(f"Mean input: {np.mean(z):1.3f}")

        
    #corr = crosscorr(np.array([z[::stepsperframe] for i in range(m)]),allsensors)
    
    #print("Sensor to filter input lag:",np.array([np.argmax(corr[i,nframes:]) for i in range(m)])*dt_real)

    zmult = np.array([z for i in range(m)])
    corr = crosscorr(zmult,allsensors,start=0.1)
    nlags = len(corr[0]) - 1
    zerolag = int(nlags/2)
    

    
    if np.std(z) == 0 or outperiod == 0:
        return None,None,None,np.max(corr)

    numinterpeaks = []
    for i in range(m):
        inpks,_ = signal.find_peaks(allintn[i])
        numinterpeaks.append(len(inpks))

    
    startframe = round(0.1*nframes)
    n_thresh = np.round(np.sum(allsensors[:,startframe:]>0,axis=1)*outperiod/(nframes-startframe)).astype(int)
    
    if plot:
       print("Number of interneuron spikes:", np.array(numinterpeaks))
       print("Number of foot triggers:", np.sum(allsensors>0,axis=1))
       print("Number of foot triggers per period:",n_thresh)
    
    lagtimes = []
    allheights = []
    mincorr = 0.0 #0.01 #0.04
    mindist = 0.05 #multiples of period
    
    
    #SENSOR-INPUT CROSS-CORRELATION
    maxlag = int(nlags/2) + outperiod
    for i in range(m):
       pks,_ = signal.find_peaks(corr[i,zerolag:maxlag],height=mincorr,distance=np.ceil(mindist*outperiod))
       if len(pks)==0: #check for edge case
           maxcorrlag = np.argmax(corr[i,zerolag:maxlag])
           if corr[i,zerolag+maxcorrlag] > mincorr:
              pks = [maxcorrlag]
              heights = [corr[i,zerolag+maxcorrlag]]
           else:
              heights = [-1]
       else:
           heights = corr[i,zerolag+pks]
       lagtimes.append(np.array(pks)*dt_real)
       allheights.append(np.mean(heights))
    
    if plot:
        plt.figure()
        plt.plot(corr[:,round(nlags/2-outperiod):round(nlags/2+outperiod)].T)
        plt.show()

    if max(allheights)<mincorr:
        return None,None,None,np.max(corr)

    if plot:
        print("Sensor to filter input lag:", lagtimes)
        print("Amplitudes:", allheights)
    
    return lagtimes, allheights, n_thresh, np.max(corr)


def crosscorr(allx, ally, start=0):
    #allx and ally must have same dimensions
    n_i = allx.shape[0]
    tt = allx.shape[1]
    tstart = round(tt*start)
    out = np.zeros([n_i, 2*(tt-tstart)-1])
    for i in range(n_i):
        x = allx[i, tstart:]
        y = ally[i, tstart:]
        if np.std(x) > 0 and np.std(y) > 0:
           out[i, :] = np.real(np.correlate(
            (x - np.mean(x))/np.std(x), (y - np.mean(y))/np.std(y), mode='full'))/len(x)

    return out

def crosscorr2(x,y,period,start=0,plot=False):
    tt = len(x)
    tstart = round(tt*start)
    x1 = x[tstart:]
    y1 = y[tstart:]
    x1 = (x1 - np.mean(x1)) / np.std(x1)
    y1 = (y1 - np.mean(y1)) / np.std(y1)
    xauto = np.correlate(x1,x1,'full')/len(x1)
    yauto = np.correlate(y1,y1,'full')/len(y1)
    
    if plot:
        plt.figure()
        plt.plot(xauto[len(x1):len(x1)+period])
        plt.plot(yauto[len(x1):len(x1)+period])
        plt.show()
    
    return np.mean((xauto[len(x1):len(x1)+period]-yauto[len(x1):len(x1)+period])**2)
    
def plotcorrs(allx,period,dt,legstrings,outfile,start=0):
    mpl.style.use('ggplot')
    tt = len(allx[0])
    tstart = round(tt*start)
    fig = plt.figure(figsize=(4,3))
    for i,x in enumerate(allx):
       x1 = x[tstart:]
       x1 = (x1 - np.mean(x1)) / np.std(x1)
       xauto = np.correlate(x1,x1,'full')/len(x1)
       plt.plot(np.array(range(period))*dt,len(allx)-i-1+xauto[len(x1):len(x1)+period])
    plt.xlabel(r'Lag time $\tau$ (s)')
    plt.ylabel(r'Autocorrelation $R_{XX}(\tau)$')
    plt.ylim([-1,1+len(allx)])
    plt.yticks([-1,0,1])
    plt.legend(legstrings)
    plt.gcf().text(0.03, 0.9,'A',fontsize=18)
    plt.tight_layout()
    plt.show()
    fig.savefig(outfile)
    



def run(env, controller, nframes, dc=0.5, tilt=0, sound=False,arousal=False,brain_in=None,footfb=None):


    dt_real,dt_cpg,t0 = UnityInterfaceBrain.gettimestep(controller.bodytype,True)
    stepsperframe = round(dt_real / dt_cpg / t0)

    m = len(controller.cpg.cons)  
    
    num_joints = 20 #unity
    action = np.zeros([1, num_joints])
    

    allbrainout = np.zeros([m,nframes])
    allintn = np.zeros([m,nframes])
    allsensors = np.zeros([m,nframes])
    allx = np.zeros([m,nframes], dtype='complex_')
    allfootfb = np.zeros([1,stepsperframe*nframes])[0]
    allfbcount = np.zeros([1,nframes])[0]
    allheight = np.zeros([1,nframes])[0]
    brain_filt = [0.0 for i in range(stepsperframe)]
    #diff = np.zeros([1,nframes])[0]

    env.reset()
    individual_name = 'RobotBehavior?team=0'
    controller_name = 'Controls?team=0'
    controlparams = []
    audio_mav = 0
    gam = 0.04    

    
    if footfb is not None:
        footdelays = [np.round(x/dt_real).astype(int) for x in footfb[0]]
        footamps = footfb[1]
        fb_thresh = footfb[2]
        n_thresh = footfb[3]
        zmean = footfb[4]
        #footcount = [0*x for x in footfb[0]]
        #training parameters:
        delta = 0.01/m # fraction of error to move
        k = 1 # window for moving average in frames
        k2 = 1 # window for gathering impulses from same foot (both sides)
        delta_th = 0.0002 #coefficient for adapting threshold
        k_th = 200 #frames for long-term moving average
        lif_gamma = 10 #leaky integrate and fire decay rate (1/s)
        #discount = []
        #for i in range(k):
        #    discount.append(1 / (i+1 - 0.5*controller.decay*dt_real*(i+1)*i))
        currthresh = fb_thresh
        max_weight = 1.5 #maximum limb weight
        thresh_tol = 1.5 #tolerance for feedback-to-input level differences
        allthreshvstime = np.nan*np.ones(nframes)
        allfootoutvstime = np.zeros([m,nframes])
        allampsvstime = np.nan*np.ones([m,nframes])
        allimpsvstime = np.zeros([sum([len(x) for x in footdelays]),nframes])

        
    burnin = 1000
    ramptime = 50           # ramping of joint activation
    audio_rampframes = 100  # ramping of audio input
    footfbstart = 100
    
    currfootfb = 0
    curract = 0
    
    #prerun the CPG to bypass transients
    for t in range(burnin):
        controller.cpg.step([0], dc, dt_cpg)
        if controller.brain is not None:
            controller.brain.step([0],0,dt_cpg)

    for t in range(nframes):
       
       obs, other = env.get_steps(individual_name)               
       if (len(obs.agent_id) == 0):
           continue

       if len(list(env._env_specs))>1:               
           contsteps, _ = env.get_steps(controller_name)
           controlparams = contsteps.obs[0]
       
       #get robot info from Unity
       currpardist,currperpdist,currheight,sidetilt,fronttilt,currtilt,orient,sensors = controller.getobs(obs.obs[0][0])
       allsensors[:,t] = sensors[0:m]
       allheight[t] = currheight
       
       ### get audio input 
       if sound and len(controlparams) > 0:
           audio_in = 20*controlparams[0][3]*min([t,audio_rampframes])/audio_rampframes
           #if debug:
           #    allaudio.append(audio_in)
           #    alltime.append(time.time())
    
           audio_mav = (1-gam)*audio_mav + gam*audio_in
           #if t % 100 == 0:
           #    print(audio_mav)
               
           if arousal: #input will drag dc level towards 0.5
               dc = dc + (0.5 - dc)*(0.5+MathUtils.sig(10*(audio_mav-0.5)))
       else:
           audio_in = 0.0

       if footfb is not None and t > footfbstart and t < nframes-k2:
           for i in range(m):
              numsens = 0
              for j in range(-k2,k2+1):
                 numsens += len(np.argwhere(allsensors[i,t+j-footdelays[i]]>0))
              if numsens>=max([1,n_thresh[i]]):
                 allfbcount[t] += 1 
                 curract = curract + max(0,footamps[i])
                 allfootoutvstime[i,t] = 1
           if curract > currthresh:
              currfootfb += 50.0*controller.decay*dt_cpg
              curract = 0
           else:
              curract = curract*(1-lif_gamma*dt_real)
           #add decay for intermediate steps
           for i in range(stepsperframe):
               allfootfb[t*stepsperframe+i] = currfootfb
               currfootfb = currfootfb*(1-controller.decay*dt_cpg)
           if brain_in is None and t>footfbstart+k_th:
               foot_longterm = np.mean(allfootfb[(t-k_th+1)*stepsperframe:(t+1)*stepsperframe:stepsperframe])
               if foot_longterm>zmean*thresh_tol:
                  currthresh += (foot_longterm-zmean*thresh_tol)*delta_th
               elif foot_longterm<zmean/thresh_tol:
                  currthresh += (foot_longterm-zmean/thresh_tol)*delta_th
               allthreshvstime[t] = currthresh
           

            

       if sound:
           brain_filt = [audio_in*controller.body.audioamp for i in range(stepsperframe)] 
       elif brain_in is not None:
           #pre-filtered from time series
           curr_brainin = brain_in[t]
           for j in range(stepsperframe):               
               brain_filt[j] = curr_brainin
               curr_brainin = curr_brainin*(1-controller.decay*dt_cpg)
           if footfb is not None and t > footfbstart and t < nframes-k2:
               #feedback training mode
               z_avg = np.mean(brain_in[(t-k+1):(t+1)])
               foot_avg = np.mean(allfootfb[(t-k+1)*stepsperframe:(t+1)*stepsperframe:stepsperframe])
               diff = z_avg - foot_avg
               #diff = brain_in[t] - allfootf,b[t*stepsperframe]
               for j in range(m):
                   for t2 in range(round(2/(lif_gamma*dt_real))):
                      if allfootoutvstime[j,t-t2] == 1:
                         footamps[j] += diff*delta*np.exp(-lif_gamma*dt_real*t2) #/allfbcount[t-i]
                   if footamps[j] > max_weight:
                      footamps[j] = max_weight
               allampsvstime[:,t] = footamps
       elif footfb is not None and t > footfbstart:
           brain_filt = allfootfb[t*stepsperframe:(t+1)*stepsperframe]
       else:
           brain_filt = [0.0 for i in range(stepsperframe)]

        
       #get CPG output
       joints_in,cpgout,brainout = controller.stepind(stepsperframe,dt_cpg,dt_real,dc,sidetilt,fronttilt,brain_filt)
       #convert to joint angles
       if t < ramptime:
            ramp = t/ramptime
       else:
            ramp = 1.0

       newactions = controller.getactions(ramp*joints_in,tilt,num_joints)



       allbrainout[:,t] = brainout[-1,:]
       allintn[:,t] = [con.x[2] for con in controller.cpg.cons]
       allx[:,t] = controller.body.limbdir[0,:]*cpgout[0,:,0] + 1j*controller.body.limbdir[1,:]*cpgout[0,:,1]

       
        
       action[0,:] = newactions
       env.set_action_for_agent(individual_name, obs.agent_id, ActionTuple(action))
       env.step()   
    
    if footfb is not None:
        zout = allfootfb
        learnout = (allampsvstime,allimpsvstime,allfootoutvstime,allthreshvstime)
    else:
        zout = brain_in
        footamps = None
        learnout = None
      
    return allbrainout[controller.body.limblist],allintn[controller.body.limblist],allx,allsensors,zout,np.mean(allheight),footamps,learnout
   
if __name__ == '__main__':
    main()
