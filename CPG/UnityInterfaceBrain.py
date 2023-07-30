import numpy as np
import matsuoka_quad
import matsuoka_hex
import matsuoka_brain
import roborun
import SimBodies
import ControllerFuncs
import math
import copy
import time
from MathUtils import sig,autocorr
from mlagents_envs.environment import UnityEnvironment
from multiprocessing import Manager
from multiprocessing.context import Process
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt

from mlagents_envs.base_env import (
    ActionTuple
)

#sine controller params (for debug)
DELTA_TIME = 0.2
AMPLITUDE = 2

num_joints = 20 #controller array size in Unity

burnin = 1000  # iterations of CPG to run before starting
ramptime = 20  # number of unity frames during which amplitude ramps up
ramp2time = 20 # number of unity frames during which leg offset transitions

#time scaling: do not change
#GLOB_MAX_T0 = 0.125
#GLOB_DEF_T0 = 0.4          #multiple of max

#time increment for CPG - does not affect dynamics
#GLOB_DT = 0.15             
#Unity parameters = time interval in build settings * frames between decisions
#GLOB_DT_UNITY = 0.1        #basic standalone
#GLOB_DT_UNITY_INT = 0.015  #interactive standalone


# Paths to the executables
def getpath(os,bodytype):
    if os=='Linux':
        paths = {'ODquad'       :r"./Unity/LinuxBuild.x86_64",
                 'shortquad'    :r"./Unity_short/LinuxShort.x86_64",
                 'AIRLhex'      :r"./Unity_hex/LinuxHex.x86_64"}
    elif os=='LinuxInt':
        paths = {'ODquad'       :r"./Unity_int/LinuxInt.x86_64",
                 'shortquad'    :r"./Unity_int/LinuxInt.x86_64",
                 'AIRLhex'      :r"./Unity_intnew/LinuxIntNew.x86_64"}
    elif os=='Windows':
        paths = {'ODquad'       :r"../My project/My project.exe",
                 'shortquad'    :r"../Short/My project.exe",
                 'AIRLhex'      :r"../Hex/My project.exe"}        
    elif os=='WindowsInt':
        paths = {'ODquad'       :r"../Interactive/My project.exe",
                 'shortquad'    :r"../Interactive/My project.exe",
                 'AIRLhex'      :r"../InteractiveNew/My project.exe"}        
    return paths[bodytype]

def gettimestep(bodytype,interactive):
    #dt_CPG is integration timestep for neurons, with time scaling factor t0
    #dt_Unity is timestep in Unity simulation

    #CPG steps per unity frame = round(dt_unity / (dt_CPG*t0))

    #steps per unity frame = 20 * t_unity / t_CPG
    if 'hex' in bodytype:
        t_unity = 0.015 #0.01
        t_CPG = 0.15 #0.1
    elif interactive:
        t_unity = 0.015
        t_CPG = 0.15
    else:
        t_unity = 0.1
        t_CPG = 0.15
    
    t0 = 0.05 #do not change    
    
    return t_unity,t_CPG,t0




class WorkerPool:
    #Queue manager for multiple simultaneous environments.
    queue = None
    
    def __init__(self, function, unitypath, port=9400, nb_workers=6, clargs=[]):
        print('Creating workers')
        self.queue = Manager().Queue()
        self.outqueue = Manager().Queue()
        self.eval = function
        self.envlist = [UnityEnvironment(file_name=unitypath, seed = 4, base_port=port, 
side_channels = [], worker_id=i, no_graphics=True, additional_args=clargs) for i in 
range(nb_workers)]
        self.processes = [Process(target=self.process,args=(self.envlist[i],)) for i in range(nb_workers)]
        for p in self.processes:
            p.start()
    def addtask(self, item):
        self.queue.put(item)
    
    def process(self,env):
        #an item should be in the form of a tuple
        #the first element is an index to identify the individual
        #the second element is the individual (forced to list)
        #the optional third element is a dictionary of kwargs
        while True:
            #wait for items if empty
            if self.queue.empty():
               continue
            item = self.queue.get()
            #None item to end the process
            if item is None:
               self.queue.task_done()
               break
            # process your item here
            if len(item)>2:
                kwargs = item[2]
            else:
                kwargs = {}
            fit = self.eval(env,list(item[1]),**kwargs)
            indnum = item[0]            
            print(item)
            print(fit)
            self.outqueue.put((indnum,fit))
            self.queue.task_done()
            
    def join(self):
        #wait until queue is empty
        self.queue.join()
        
    def terminate(self):
        """ wait until queue is empty and terminate processes """
        self.queue.join()
        for env in self.envlist:
            env.close()
        for p in self.processes:
            p.terminate()






def getlimbcorr(allx,start=0.5):
    mincorr = 0.2
    tt = allx.shape[1]
    k  = allx.shape[0]
    tstart = round(tt*start)
    dx = np.zeros([k,tt-tstart-1])
    stdx = [0 for i in range(k)]
    corr = np.zeros([k,k])
    for i in range(k):
        dx[i,:] = np.diff(np.real(allx[i,tstart:]))
        stdx[i] = np.std(np.real(allx[i,tstart:]))
    for i in range(k):
        for j in range(i+1,k):
            if stdx[i]==0 or stdx[j]==0:
               corr[i,j] = 0
            else:
               coeffs = np.corrcoef(dx[i,:],dx[j,:])
               corr[i,j] = coeffs[0,1]
    
    maxcorr = np.max(corr.flatten())
    maxind = np.argmax(corr.flatten())
    graph = corr>mincorr
    n_components = connected_components(csgraph=graph, directed=True, return_labels=False)

    return maxcorr, maxind, n_components

def evaluate(env, individual, pint, bodytype, dc_in=[0.5, 0.5], brain=None, brain_in=None, outw=None, outbias=None, tilt_in=None, decay=1, use_controller=True, getperiod=False, timeseries=False, interactive=False, arousal=False, sound=False, nframes=100, maxperiod=-1):
    #tilt_in (if not None) must be one less than length of dc_in   
    #interactive refers to the LinuxInt/WindowsInt standalone (can be run in automatic mode with nframes>0)

    
    debug = False

    if 'hex' in bodytype:
        body = SimBodies.HexBody(bodytype,pint)
    else:
        body = SimBodies.QuadBody(bodytype,pint)

    
    controller = ControllerFuncs.Controller(body,individual,brain,outw,decay,outbias,bodytype)
    
    if nframes < 0:
        interactive=True
        nstages = 1
    else:
        nstages = len(dc_in)-1

    # dtUnity = time interval of the unity player in seconds
    dtUnity,dt,t0 = gettimestep(bodytype,interactive)
    # dt = CPG time step
    # t0 = seconds per CPG time step
    #t0 = GLOB_MAX_T0 * GLOB_DEF_T0 
    stepsperframe = round(dtUnity / dt / t0)

    k = len(individual.cons) #number of limbs


        
    controlparams = []
    tilt_mode = 1 #0=body tilt, 1=leg offset


    if tilt_in == None:
        tilt_in = [0 for i in range(nstages)]

    stagex = [0 for i in range(nstages+1)]
    stagez = [0 for i in range(nstages+1)]
    stageorient = [0 for i in range(nstages+1)]

    pardist = [0 for i in range(nstages)]
    perpdist = [0 for i in range(nstages)]
    heading = [0 for i in range(nstages)]


    ##hip feedback
    #hipfeedback = False
    #hip_fb_amp = -0.05  # <0
    #gam1 = 0.05
    #hip_int = 0

    #if interactive:
    #    time.sleep(0.05)

    env.reset()
   

    individual_name = 'RobotBehavior?team=0' #list(env._env_specs)[0]
    controller_name = 'Controls?team=0'
    #print(list(env._env_specs))
    action = np.zeros([1, num_joints])
    #A_fb = np.array([0 for i in range(k)])
    #B_fb = np.array([0 for i in range(k)])
    #brain_out = np.array([0 for i in range(k)])
    tilttot = [0 for i in range(nstages)]
    tiltcount = [0 for i in range(nstages)]
    heighttot = [0 for i in range(nstages)]
    heightmean = [0 for i in range(nstages)]
    tiltmean = [0 for i in range(nstages)]

    if debug:
        allaudio = []
        alltime = []

    if getperiod or timeseries:
        allx = np.zeros([k,nstages*nframes*stepsperframe], dtype='complex_')
        if brain is not None:
            allbrain = np.zeros([k,nstages*nframes*stepsperframe])

    if timeseries:
        alltilt = np.zeros([1,nstages*nframes]).flatten()
        allheight = np.zeros([1,nstages*nframes]).flatten()
        allpardist = np.zeros([1,nstages*nframes]).flatten()
        allperpdist = np.zeros([1,nstages*nframes]).flatten()
        
    #prerun the CPG to bypass transients
    for m in range(burnin):
        #out = individual.step([0], dc_in[0], dt)
        individual.step([0], dc_in[0], dt)
        if brain is not None:
            brain.step([0],0,dt)
        #oldpos = np.array([hinge(x) for x in out])
    
    brain_filt = 0
    audio_mav = 0
    gam = 0.04
    
    audio_rampframes = 100
    
    j = -1
    while True:
        
        j += 1
        if nframes>0 and j >= nstages*nframes:
            break
        
        obs, other = env.get_steps(individual_name)
        # obs[0][0]:
        # elements 0-2 = position x,y,z
        # elements 3-5 = vector normal to top x,y,z
        # elements 6-8 = vector normal to front x,y,z
        # y = vertical, z=(initial) forward dir
        
        #print(list(env._env_specs))
        if len(list(env._env_specs))>1:               
               contsteps, _ = env.get_steps(controller_name)
               controlparams = contsteps.obs[0]
               #print(controlparams.obs[0])
               
        if (len(obs.agent_id) > 0):
            if nframes>0:
               stage = j // nframes
            else:
               stage = 0

            ### calculate control parameters
            if nframes>0:
                dc = dc_in[stage] + (dc_in[stage+1]-dc_in[stage])*(j % nframes)/nframes
                
                if tilt_mode == 0:
                   tilt = tilt_in[stage]
                   tilt2 = 0
                else:
                   tilt = 0
                   if stage>0 and j % nframes < ramp2time:
                      tilt2 = tilt_in[stage-1] + (tilt_in[stage]-tilt_in[stage-1])*(j % nframes)/ramp2time
                   else:
                      tilt2 = tilt_in[stage]
                
            elif len(controlparams) > 0:
                if controlparams[0][0] < -100:
                    print("Exiting")
                    break
                elif controlparams[0][0] < 0:
                    print("Resetting")
                    env.reset()
                    time.sleep(0.1)
                    j = -1
                    brain_filt = 0
                    continue
                tilt2 = controlparams[0][1]
                dc = controlparams[0][2]
                   
                tilt = 0.0 #target tilt for body stabilization
            else:
                dc = dc_in[0]
                tilt2 = 0.0
                audio_in = 0.0
                tilt = 0.0
            
            ### get audio input 
            if sound and len(controlparams) > 0:
                audio_in = 20*controlparams[0][3]*min([j,audio_rampframes])/audio_rampframes

                if debug:
                   allaudio.append(audio_in)
                   alltime.append(time.time())

                audio_mav = (1-gam)*audio_mav + gam*audio_in
                if j % 100 == 0:
                   print(audio_mav)
                   
                if arousal: #input will drag dc level towards 0.5
                    dc = dc + (0.5 - dc)*(0.5+sig(10*(audio_mav-0.5)))
            else:
                audio_in = 0.0
            
            
            ### get observations
            
            currpardist,currperpdist,currheight,sidetilt,fronttilt,currtilt,orient,_,_,_ = controller.getobs(obs.obs[0][0])

            tilttot[stage] = tilttot[stage] + currtilt
            heighttot[stage] = heighttot[stage] + currheight
            tiltcount[stage] = tiltcount[stage] + 1
            
            if timeseries:
                allheight[stage*nframes+j] = currheight
                alltilt[stage*nframes+j] = currtilt
                allpardist[stage*nframes+j] = currpardist
                allperpdist[stage*nframes+j] = currperpdist
                
            ### start-of-stage measurements
            
            if j % nframes == 0:
                stagez[stage] = currpardist
                stagex[stage] = currperpdist
                stageorient[stage] = orient

            ### step CPG x stepsperframe

            if (use_controller):
                
                if brain_in is not None:
                    #pre-filtered from time series
                    brain_filt = brain_in[j*stepsperframe:(j+1)*stepsperframe]
                else:
                    brain_filt = [audio_in*body.audioamp for i in range(stepsperframe)] 

                
                joints_in,cpg_out,brain_out = controller.stepind(stepsperframe,dt,dtUnity,dc,sidetilt,fronttilt-tilt,brain_filt)

                if j < ramptime:
                    ramp = j/ramptime
                else:
                    ramp = 1
                
                #allow ramping up from zero
                joints_in_r = ramp*joints_in
                
                if getperiod:
                    for m in range(stepsperframe):
                       allx[:, j*stepsperframe+m] = body.limbdir[0,:]*cpg_out[m,:,0] + 1j*body.limbdir[1,:]*cpg_out[m,:,1]
                       allbrain[:, j*stepsperframe+m] = brain_out[m,:]   
                        
                newactions = controller.getactions(joints_in_r,tilt2,num_joints)
                
                action[0,:] = newactions

            else:
                # sine wave actions (old code, may not work)
                for i in range(len(action[0])):
                    action[0, i] = np.sin(j * DELTA_TIME) * AMPLITUDE
            # print(obs.agent_id)
            env.set_action_for_agent(individual_name, obs.agent_id, ActionTuple(action))
            env.step()


    #measurements
    if len(obs) > 0:

        stage += 1
        stagez[stage] = obs.obs[0][0][2]
        stagex[stage] = obs.obs[0][0][0]
        stageorient[stage] = math.atan2(obs.obs[0][0][6], obs.obs[0][0][8])

        for i in range(nstages):
            heading[i] = math.atan2(stagex[i+1]-stagex[i], stagez[i+1]-stagez[i])
            dist = math.sqrt((stagex[i+1]-stagex[i])**2 + (stagez[i+1]-stagez[i])**2)
            pardist[i] = dist*math.cos(stageorient[i]-heading[i])
            perpdist[i] = dist*math.sin(stageorient[i]-heading[i])


        heightmean = [heighttot[i]/tiltcount[i]/body.ymax for i in range(nstages)]
        tiltmean = [tilttot[i]/tiltcount[i] for i in range(nstages)]

        
    if getperiod:
        mindelay = 1  # seconds?
        if maxperiod < 0:
            maxdelay = -1
        else:
            maxdelay = round(maxperiod/dt)
        #get autocorrelation peak in CPG steps
        peaks, heights = autocorr(allx, 0.33, round(mindelay/dt), maxdelay=maxdelay)
        #convert to simulation seconds
        period = peaks*dtUnity/stepsperframe #dt*peaks
        corr,corrind,n_comp = getlimbcorr(allx)
        #plt.plot(allbrain.T)
        #plt.show()
    else:
        period = None
        corr = None
        corrind = None
        n_comp = None


    if debug:
        timearr = np.array(alltime)-alltime[0]
        plt.plot(timearr,allaudio)
        plt.waitforbuttonpress()
        plt.plot(np.diff(timearr))
        #plt.hist(np.diff(timearr),bins=20)
        plt.waitforbuttonpress()

    
    if 'hex' in bodytype:
        corrout = n_comp
    else:
        corrout = corrind    

    if timeseries:
        return (allpardist, allperpdist, allheight/body.ymax, alltilt, allx)
    else:
        return (pardist, perpdist, heightmean, tiltmean, period, corr, corrout)



def iterate(n, dc_arr, tilt_arr, bodytype, env, ind, iterations=3, seed=111):

    
    meandist = np.zeros([len(dc_arr), len(tilt_arr)])
    meanperiod = np.zeros([len(dc_arr), len(tilt_arr)])
    meanheight = np.zeros([len(dc_arr), len(tilt_arr)])
    meantilt = np.zeros([len(dc_arr), len(tilt_arr)])
    meancorr = np.zeros([len(dc_arr), len(tilt_arr)])
    meancorrind = np.zeros([len(dc_arr), len(tilt_arr)])

    # warmup
    if 'hex' in bodytype:
        rob = matsuoka_hex.array2param(ind[:n])
        kwargs = {'nframes':1000}
    else:
        rob = matsuoka_quad.array2param(ind[:n])
        kwargs = {}
        
    evaluate(env, rob, ind[n:], bodytype)
    print('(warm up)')

    for i, dc in enumerate(dc_arr):
        for j, tilt in enumerate(tilt_arr):
            totdist = []
            totperiod = []
            totheight = []
            tottilt = []
            totcorr = []
            totcorrind = []

            print(dc, tilt)

            for k in range(iterations):
                rob.reset(seed=seed+k)
                (pardist, perpdist, heightmean, tiltmean, period, corr, corrind) = evaluate(
                    env, rob, ind[n:], bodytype, dc_in=[dc, dc, dc, dc], tilt_in=[tilt,tilt,tilt], getperiod=True, **kwargs)
                print(sum(pardist), period, heightmean[-1])
                totdist.append(np.mean(pardist))
                totperiod.append(period)
                totheight.append(heightmean[-1])
                tottilt.append(np.mean(tiltmean))
                totcorr.append(corr)
                totcorrind.append(corrind)
                
            meandist[i, j] = np.median(totdist)
            meanperiod[i, j] = np.median(totperiod)
            meanheight[i, j] = np.median(totheight)
            meantilt[i, j] = np.median(tottilt)
            meancorr[i, j] = np.median(totcorr)
            meancorrind[i,j] = np.median(totcorrind)

    return meandist, meanperiod, meanheight, meantilt, meancorr, meancorrind


def run_from_array(n, bodytype, env, p, dc=[1,0.5,0.5,1], tilt=[-0.015,0.015,0.015], warmups=1, k=3, nograph=True, seed=None):

    env.reset()

    nobj = 4  # number of objectives

    fittot = [[] for i in range(nobj)]

    if 'hex' in bodytype:
       x0 = np.sqrt(5)
       z0 = 2.5
       rob = matsuoka_hex.array2param(p[:n])
       kwargs = {"nframes":1000}
    else:            
       x0 = np.sqrt(5)  # scaling for lateral movement punishment
       z0 = 2.5  # optimal distance for second stage
       rob = matsuoka_quad.array2param(p[:n])
       kwargs = {}
    
    outlines = []

    for i in range(warmups):
        rob.reset(seed)
        evaluate(env, rob, p[n:], bodytype, dc_in=dc)
        if not nograph:
            outlines.append('(warm up)')
        
    for i in range(k):
        newfits = [0 for i in range(nobj)]
        j = 0
        while newfits[0] == 0:
            j += 1
            if j > 5:
                print('5 evaluations with no fitness')
                break
            
            if seed is not None:
               rob.reset(seed+i)
            else:
               rob.reset()
            (pardist, perpdist, heightmean, tiltmean, _, _, _) = evaluate(env, rob, p[n:], bodytype, dc_in=dc, tilt_in=tilt, 
**kwargs)
            #define fitnesses here
            newfits[0] = (0-pardist[0]) - (perpdist[0]/x0)**2
            newfits[1] = 2*z0*pardist[1] - pardist[1]**2 - (perpdist[1]/x0)**2
            newfits[2] = pardist[2] - (perpdist[2]/x0)**2
            newfits[3] = z0*z0*min([1,np.mean(heightmean)]) / (1 + np.mean(tiltmean))

            if not nograph:
                outlines.append(' '.join([str(fit) for fit in newfits]))
        for i in range(nobj):
            fittot[i].append(newfits[i])

    if not nograph:
        for line in outlines:
            print(line)

    #print(ind_num, 'done')

    fitout = [np.median(arr) for arr in fittot]

    return tuple(fitout)


def run_with_input(env,cpg,body_inds,bodytype,baseperiod,brain,outw,decay,outbias,t_arr,amp,nframes,dc,tilt,graphics=False,skipevery=-1,sdev=0,tstart=0,tend=1,timeseries=False,seed=None,asym=None,**kwargs):
    #assumes baseperiod is in simulation seconds
    
    alldist = []
    allheight = []
    allperiod = []
    allperpdist = []
    alloutput = []
    allinput = []
    alltilt = []

    dtUnity,dt,t0 = gettimestep(bodytype,False)
    # dt_unity = time interval of the unity player
    # dt = time interval of the simulation

    #t0 = GLOB_MAX_T0 * GLOB_DEF_T0
    stepsperframe = round(dtUnity / dt / t0)
    tcon = stepsperframe*dt/dtUnity #CPG to unity timescale conversion (~t0)
    tt = stepsperframe*nframes
    
    max_r = 4.5
    
    #first run with no input
    if brain==None:
        zero_std = 0
    else:
        newcpg = copy.deepcopy(cpg)
        newbrain = copy.deepcopy(brain)
        z = np.zeros([tt+1,1])
        firstsim = roborun.Sim(newcpg,z[:,0],[dc],dt=dt,brain=newbrain,outw=outw,outbias=outbias)
        firstsim.run()
        zero_std = np.mean(np.std(firstsim.brainx[:,int(tt/2):],1))
        del newcpg
        del newbrain
    
    newcpg = copy.deepcopy(cpg)
    #run Unity with no input
    evaluate(env, newcpg, body_inds, bodytype, dc_in = [dc,dc])
    del newcpg

    for i,t in enumerate(t_arr):
        if seed==None:
            seed1 = None
            seed2 = None
            seed3 = None
        else:
            seed1 = seed+3*i
            seed2 = seed+3*i+1
            seed3 = seed+3*i+2

        
        z = 100*amp*roborun.periodinput(t*tcon*baseperiod,int(tt*tstart),int(tt*tend),tt,dt,skipevery=skipevery,sdev=sdev,seed=seed3,asym=asym)
        z = roborun.rc_lpf(z,decay*dt)
        maxperiod = max_r*t*tcon*baseperiod/2

        newcpg = copy.deepcopy(cpg)
        newcpg.reset(seed1)
        newbrain = copy.deepcopy(brain)
        if newbrain is not None:
            newbrain.reset(seed2)
        if timeseries:
            (pardist, perpdist, height, tilt, output) = evaluate(env, newcpg, body_inds, bodytype, dc_in = [dc,dc], tilt_in=[tilt], brain=newbrain, brain_in=z, outw=outw, outbias=outbias, getperiod=True, nframes=nframes, maxperiod=maxperiod, timeseries=True, **kwargs)
            alldist.append(pardist)
            allheight.append(height)
            alltilt.append(tilt)
            allperpdist.append(perpdist)
            alloutput.append(output)
            allinput.append(z)
        else:
            (pardist, perpdist, heightmean, tiltmean, period, _, _) = evaluate(env, newcpg, body_inds, bodytype, dc_in = [dc,dc], tilt_in=[tilt], brain=newbrain, brain_in=z, outw=outw, outbias=outbias, getperiod=True, nframes=nframes, maxperiod=maxperiod, **kwargs)
            alldist.append(pardist[0])
            allheight.append(heightmean[0])
            allperiod.append(period)
        del newcpg
        del newbrain

    if timeseries:
        outputs = (alldist,allperpdist,allheight,alltilt,allinput,alloutput)
    else:   
        outputs = (alldist,allheight,allperiod,zero_std)
        
    return outputs

def run_brain_array(n_brain,cpg,body_inds,baseperiod,bodytype,env,inds,ratios=[0.618,1,1.618],dc=0.5,tilt=0,numiter=1,combined=True,graphics=False,**kwargs):
    #called by genetic algorithm. calculates and returns score only

    t_arr = []
    amp = 1.0
    if 'hex' in bodytype:
        nframes = 4000
    else:
        nframes = 400
    std_epsilon = 0.1
    tdiff_epsilon = 0.1
    max_r = 4.5
    allr = []
    
    for ratio in ratios:
        for i in range(numiter):
            t_arr.append(ratio)

    m = len(cpg.cons)
    brain,outw,decay,outbias = matsuoka_brain.array2brain(n_brain,m,inds)

    (alldist,allheight,allperiod,zero_std) = run_with_input(env,cpg,body_inds,bodytype,baseperiod,brain,outw,decay,outbias,t_arr,amp,nframes,dc,tilt,graphics=graphics,**kwargs)

    fits = []
    for i in range(len(t_arr)):
        r = 2*allperiod[i]/t_arr[i]/baseperiod
        allr.append(r)
        if r == 0 or r > max_r:
            pscore = 0
        else:
            tdiff = abs(round(r) - r)
            pscore = 1/(1 + tdiff/tdiff_epsilon + zero_std/std_epsilon)
        if combined:
            fits.append(min([1,allheight[i]])*pscore)
        else:
            fits.append(allheight[i])
            fits.append(pscore)
    if graphics:
        print(allheight)
        print(allperiod)
        print(allr)
        print(zero_std)

    #print(str(ind_num) + ' done')

    return tuple(fits)



if __name__ == "__main__":
    n = 23  # number of CPG parameters
    m = 10  # physical parameters
    n_brain = 6 # number of filter neurons
    
    #CPG array
    p = [1, 10, 1, 2, 8, 5, 5, 3, 10, 10, 5, 9, 8, 8, 9, 6, 4, 9, 1, 2, 8, 8, 5, 9, 10, 7, 2, 10, 2, 1, 1, 2]
    #filter array
    b = [2, 5, 10, 5, 9, 8, 3, 9, 1, 6, 6, 1, 1, 2, 1, 6, 8, 8, 7, 8, 5, 3, 8, 3, 1, 7, 9, 9, 8, 9, 9, 8, 6, 7, 9, 5, 3, 7, 10, 3, 2, 7, 10, 2, 6, 9, 7, 6, 10, 6, 4, 9, 9, 8, 5, 6, 4, 3, 5, 5, 6, 1]
    baseperiod = 8.16 #cpg units
    
    t0 = 0.0521
    tmult = t0 #set to 1 if baseperiod is in seconds
    
    short = True
    osys = 'Windows'
    
    #function 1: CPG evaluation, function 2: CPG+filter evaluation
    function = 2
    
    if short:
        bodytype = 'shortquad'
    else:    
        bodytype = 'ODquad'

    Path = getpath(osys,bodytype)
    env = UnityEnvironment(file_name=Path, seed=4, side_channels=[], worker_id=0, no_graphics=False, additional_args=['-nolog'])

    if function==1:
        output = run_from_array(n, bodytype, env, p, nograph=False, seed=111)
        print(output)

    if function==2:
        cpg = matsuoka_quad.array2param(p[:n])
        body_inds = p[n:]
        output = run_brain_array(6,cpg,body_inds,baseperiod*tmult,bodytype,env,b,graphics=True,skipevery=4,sdev=0.02,seed=111)
        print(output)
     
