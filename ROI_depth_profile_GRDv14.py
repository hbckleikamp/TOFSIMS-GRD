# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 11:38:30 2024

@author: hkleikamp
"""


#%% Modules


import pySPM
import numpy as np
import pandas as pd
import struct

from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import scipy
from scipy.signal import savgol_filter
from scipy.signal import fftconvolve
from scipy.optimize import least_squares
from scipy.signal import find_peaks



from bisect import bisect_left #Fast pick close

from functools import reduce
#%% Disclaimer
#the type of material this script is written for is specific.
    #-inhomogenous height (local ppm correction)
    #-homogenous space (only 2 ROIs)

#base calibrants
Calibrants=["Na+","K+","Ca+","CH2+","CH3+","C2H3+","C3H5+","C4H7+","C6H5+","C4H7O+","C5H11+","C8H19PNO4+","C27H45+","C27H45O+","C29H50O2+", #positive calibrants
            "C2-","C3-","CH-","CH2-","C2H4-","C4H-","C16H31O2−","C18H33O2−","C18H35O2−","C27H45O−","C29H49O2−"                           #negative calibrants
            ,"S-"
            
            ]

#Fiber specific calibrants
cfile="C:/Wout_features/Sheath_calibrants.tsv"
csd=pd.read_csv(cfile,sep="\t")
exCalibrants=csd["ToF-SIMS assignment"].tolist() #extended calibrants (used for 3D ppm calibration)

emass=0.000548579909 #mass electron
elements=pd.read_csv("C:/Wout_features/utils/natural_isotope_abundances.tsv",sep="\t") #Nist isotopic abundances and monisiotopic masses
elements=elements[elements["Standard Isotope"]].set_index("symbol")["Relative Atomic Mass"]
elements=pd.concat([elements,pd.Series([-emass,emass],index=["+","-"])]) #add charges


#%%



bin_tof=1 #3
bin_pixels=1 #3 #bins in 2 directions: 2->, 3->9 data reduction
bin_scans=10 #only used for ROI detection and local calibration
ppm_cal=350

#peak picking
smoothing_window=int(30/bin_tof) #window of rolling mean smoothing prior to peak picking
max_width=2000 #on both sides (so actually *2)
prominence=3 #minimum prominence
distance=20  #minimum distance between peaks=
top_peaks=100 #number of top peaks used for calibration and ROI detection
Dead_time_corr=True #False #True
remove_corner=False

Calibrate_global=exCalibrants #False #exCalibrants #Calibrants
Calibrate_local=False #only useful if samples have very inhomogenous heigt (single fibrils). exCalibrants #False #Calibrants #Calibrants #exCalibrants


#Kmeans
n_clusters=2 
break_percentage=0.05 #stop analyzing scans after the amount of analyzed masses has not change by this amount (sample steady state)
max_scans=False #50 #stop analyzing scans after this number (mostly for testing)

min_count=3 #data reduction for spectra storage



# Delayed extraction needs higher mass ions
# Green, F. M., Ian S. Gilmore, and Martin P. Seah. "TOF-SIMS: Accurate mass scale calibration." Journal of the American Society for Mass Spectrometry 17 (2006): 514-523.
# Vanbellingen, Quentin P., et al. "Time‐of‐flight secondary ion mass spectrometry imaging of biological samples with delayed extraction for high mass and high spatial resolutions." Rapid Communications in Mass Spectrometry 29.13 (2015): 1187-1195.
### Olivier Scholder. (2018, November 28). scholi/pySPM: pySPM v0.2.16 (Version v0.2.16). Zenodo. http://doi.org/10.5281/zenodo.998575
#https://github.com/scholi/pySPM/blob/master/pySPM/ITM.py


#%% Functions

def residual(p,x,y):
    return (y-c2m(x,*p,bin_tof=bin_tof))/y

def m2c(m,sf,k0,bin_tof=bin_tof): #mass 2 channel
    return np.round(  ( sf*np.sqrt(m) + k0 )  / bin_tof  ).astype(int)

def c2m(c,sf,k0,bin_tof=bin_tof):
    return ((bin_tof * c - k0) / (sf)) ** 2   

##fast take closest 
# https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value
def take_closest(myList, myNumber):
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before
    
def parse_form(form): #chemical formular parser
    e,c,comps="","",[]
    for i in form:
        if i.isupper(): #new entry   
            if e: 
                if not c: c="1"
                comps.append([e,c])
            e,c=i,""         
        elif i.islower(): e+=i
        elif i.isdigit(): c+=i
    if e: 
        if not c: c="1"
        comps.append([e,c])
    
    cdf=pd.DataFrame(comps,columns=["elements","counts"]).set_index("elements").T.astype(int)
    cdf["+"]=form.count("+")
    cdf["-"]=form.count("-")
    return cdf

def getMz(form): #this could be vectorized for speed up in the future
    cdf=parse_form(form)
    return (cdf.values*elements.loc[cdf.columns].values).sum() / cdf[["+","-"]].sum(axis=1)
    
def savgol2d( z, window_size, order):
    #https://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
    n_terms = ( order + 1 ) * ( order + 2)  / 2.0
    
    if  window_size % 2 == 0:    raise ValueError('window_size must be odd')
    if window_size**2 < n_terms: raise ValueError('order is too high for the window size')

    half_size = window_size // 2
    exps = [ (k-n, n) for k in range(order+1) for n in range(k+1) ]
    
    # coordinates of points
    ind = np.arange(-half_size, half_size+1, dtype=np.float64)
    dx = np.repeat( ind, window_size )
    dy = np.tile( ind, [window_size, 1]).reshape(window_size**2, )

    # build matrix of system of equation
    A = np.empty( (window_size**2, len(exps)) )
    for i, exp in enumerate( exps ):
        A[:,i] = (dx**exp[0]) * (dy**exp[1])
        
    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2*half_size, z.shape[1] + 2*half_size
    Z = np.zeros( (new_shape) )
    Z[:half_size,  half_size:-half_size] =  z[0, :]  -  np.abs( np.flipud( z[1:half_size+1, :] )  - z[0,  :] )  # top band
    Z[-half_size:, half_size:-half_size] =  z[-1, :]  + np.abs( np.flipud( z[-half_size-1:-1, :] ) -z[-1, :] )  # bottom band
    
    band = np.tile( z[:,0].reshape(-1,1), [1,half_size])                                                        # left band
    Z[half_size:-half_size, :half_size] = band - np.abs( np.fliplr( z[:, 1:half_size+1] ) - band )
    band = np.tile( z[:,-1].reshape(-1,1), [1,half_size] )                                                      # right band
    Z[half_size:-half_size, -half_size:] =  band + np.abs( np.fliplr( z[:, -half_size-1:-1] ) - band )
    
    Z[half_size:-half_size, half_size:-half_size] = z                                                           # central band
    Z[:half_size,:half_size] = z[0,0] - np.abs( np.flipud(np.fliplr(z[1:half_size+1,1:half_size+1]) ) - z[0,0] )# top left corner
    Z[-half_size:,-half_size:] = z[-1,-1] + np.abs( np.flipud(np.fliplr(z[-half_size-1:-1,-half_size-1:-1]) ) - z[-1,-1] )  # bottom right corner
    
    band = Z[half_size,-half_size:]                                                                             # top right corner
    Z[:half_size,-half_size:] = band - np.abs( np.flipud(Z[half_size+1:2*half_size+1,-half_size:]) - band ) 
    band = Z[-half_size:,half_size].reshape(-1,1)                                                               # bottom left corner
    Z[-half_size:,:half_size] = band - np.abs( np.fliplr(Z[-half_size:, half_size+1:2*half_size+1]) - band ) 
    

    m = np.linalg.pinv(A)[0].reshape((window_size, -1))
    return scipy.signal.fftconvolve(Z, m, mode='valid')

    

#%%

itmfiles=["E:/Antwerp/ToF-SIMS Louvain/20240117_cablebacteria_backup/N_Slide I spot VIII.itm",
"E:/Antwerp/ToF-SIMS Louvain/20240117_cablebacteria_backup/N_Slide I spot IV.itm",
"E:/Antwerp/ToF-SIMS Louvain/20240117_cablebacteria_backup/N_Slide I spot VIIa.itm",
"E:/Antwerp/ToF-SIMS Louvain/20240117_cablebacteria_backup/N_Slide I spot VIIb.itm",
"E:/Antwerp/ToF-SIMS Louvain/20240131_CableBacteria/N_cableBacteria for XPS.itm",
"E:/Antwerp/ToF-SIMS Louvain/20240131_CableBacteria/N_Oregon.itm",
"E:/Antwerp/ToF-SIMS Louvain/20240131_CableBacteria/N_Slide I spot II.itm",
# "E:/Antwerp/ToF-SIMS Louvain/20240131_CableBacteria/N_Slide I spot III.itm",
# "E:/Antwerp/ToF-SIMS Louvain/20240131_CableBacteria/N_Slide I spot III_2ndpart.itm",
"E:/Antwerp/ToF-SIMS Louvain/20240131_CableBacteria/N_Slide II spot III.itm",
"E:/Antwerp/ToF-SIMS Louvain/20240131_CableBacteria/N_Slide II spot IV.itm",
"E:/Antwerp/ToF-SIMS Louvain/20240131_CableBacteria/N_Slide II spot IX.itm",
"E:/Antwerp/ToF-SIMS Louvain/20240131_CableBacteria/N_Slide II spot V.itm",
"E:/Antwerp/ToF-SIMS Louvain/20240131_CableBacteria/N_Slide II spot VI.itm",
"E:/Antwerp/ToF-SIMS Louvain/20240131_CableBacteria/N_Slide II spot VII.itm",
"E:/Antwerp/ToF-SIMS Louvain/20240131_CableBacteria/N_Slide II spot VIII.itm",
"E:/Antwerp/ToF-SIMS Louvain/20240131_CableBacteria/N_Slide III spot II.itm",
"E:/Antwerp/ToF-SIMS Louvain/20240131_CableBacteria/N_Slide III spot III.itm",
"E:/Antwerp/ToF-SIMS Louvain/20240131_CableBacteria/N_Slide III spot IV.itm",
"E:/Antwerp/ToF-SIMS Louvain/20240131_CableBacteria/N_Slide III spot VI.itm",
"E:/Antwerp/ToF-SIMS Louvain/20240131_CableBacteria/N_Slide III spot VII.itm",
"E:/Antwerp/ToF-SIMS Louvain/20240131_CableBacteria/N_Slide III spot VIII.itm",
"E:/Antwerp/ToF-SIMS Louvain/20240131_CableBacteria/P_Oregon.itm",
"E:/Antwerp/ToF-SIMS Louvain/20240131_CableBacteria/P_Slide I spot VI.itm"]

#testing files
#itmfiles=["E:/Antwerp/ToF-SIMS Louvain/20240131_CableBacteria/N_Slide I spot II.itm"]
#itmfiles=["E:/Antwerp/ToF-SIMS Louvain/20240131_CableBacteria/P_Oregon.itm"]
#itmfiles=["E:/Antwerp/ToF-SIMS Louvain/20240131_CableBacteria/P_Slide I spot VI.itm"]
#itmfiles=["E:/Antwerp/ToF-SIMS Louvain/20240131_CableBacteria/N_Slide III spot VIII.itm"]
# #itmfiles=["E:/Antwerp/ToF-SIMS Louvain/20240131_CableBacteria/P_Slide I spot VI.itm"]
#itmfiles=["E:/Antwerp/ToF-SIMS Louvain/20240117_cablebacteria_backup/N_Slide I spot IV.itm"]
#custom mass calc

#itmfiles=["E:/Antwerp/ToF-SIMS Louvain/20240131_CableBacteria/N_cableBacteria for XPS.itm"]

grd_exe="C:/Program Files (x86)/ION-TOF/SurfaceLab 6/bin/ITRawExport.exe"

import subprocess
import os
for itmfile in itmfiles:
    
    grdfile=itmfile+".grd"
    if not os.path.exists(grdfile):
        command='"'+grd_exe+'"'+' "'+itmfile+'"'
        print(command)
        stdout, stderr =subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    

    #%% load ITM
    
    
    I = pySPM.ITM(itmfile)
    
    ### Get metadata ###
    
    xpix,ypix= I.size["pixels"]["x"], I.size["pixels"]["y"]
    channel_width=I.get_value("Registration.TimeResolution")["float"]
    dead_time=I.get_value("Context.Mode.StaticSIMS.TDC.DeadTime")["float"]
    k0,sf=I.k0,I.sf
    mode=I.polarity #negative or positive
    mode_sign="-"
    if "ositive" in mode: mode_sign="+"
    scans=I.root.goto("propend/Measurement.ScanNumber").get_key_value()["int"]
    number_channels = int(round(I.root.goto('propend/Measurement.CycleTime').get_key_value()['float'] \
                                / I.root.goto('propend/Registration.TimeResolution').get_key_value()['float'])/bin_tof)
    nan_arr=np.array([np.nan]*number_channels)    
    dt=int(round(dead_time/channel_width,0)/bin_tof)
    
    #check values copy for excel
    #vals=pd.Series(I.getValues()).explode().str.replace("\x00"," ").reset_index().astype(str).to_clipboard()


    ### Unpack raw scans ###
    
    with open(grdfile,"rb") as fin:
        ds=np.fromfile(fin,dtype=np.uint32).reshape(-1,5)[:,[0,2,3,4]]
    ds=np.hstack([ds,np.ones(len(ds)).reshape(-1,1)]) #add weights

    ### Binning ###
    
    ds[:,1:3]=(ds[:,1:3]/bin_pixels).astype(int)
    ds[:,3]=  (ds[:,3]  /bin_tof   ).astype(int) 
 
    if remove_corner: #prevents corner artifacts from pixel binning 
        ds=ds[(ds[:,1]>ds[:,1].min()) & (ds[:,1]<ds[:,1].max()) & (ds[:,2]>ds[:,2].min()) & (ds[:,2]<ds[:,2].max())]
     
    ### calculate peaks ###
    
    u,uc=np.unique(ds[:,3],return_counts=True)#,return_inverse=True)
    Spectrum=nan_arr
    Spectrum[u.astype(int)]=uc
    Spectrum=pd.Series(Spectrum).interpolate().fillna(0).rolling(window=smoothing_window).mean()  #rolling mean will cause slight shifts in peak location.
    p,pi=find_peaks(Spectrum,prominence=prominence,distance=distance) 
    tp=p[np.sort(np.argsort(pi["prominences"])[-top_peaks:])]
    
    ### Dead time correction ###
    
    if Dead_time_corr: #T. Stephan, J. Zehnpfenning and A. Benninghoven, J. vac. Sci. A 12 (2), 1994
        N=xpix*ypix*scans  
        Np = np.zeros(Spectrum.shape)
        Np = (N- np.convolve(Spectrum, np.ones(dt - 1, dtype=int), "full")[: -dt + 2])
        Np[Np == 0] = 1
        dSpectrum = -N * np.log(1 - Spectrum / Np)
        ds[:,4]=(dSpectrum/Spectrum)[ds[:,3]]
        
        ds[:,4][~np.isfinite(ds[:,4])]=1
        ds[ds[:,4]<1,4]=1

    #### Global calibration ###
    #%%
    if Calibrate_global:
      
        k0,sf=I.k0,I.sf #test
        fCalibrants=np.sort(np.array([pySPM.utils.get_mass(i) for i in Calibrate_global if i.endswith(mode_sign)])) #shortended calibrants list
        mp=c2m(tp,sf,k0)

        #get calibrants
        ppms=abs((fCalibrants-mp.reshape(-1,1))/mp.reshape(-1,1))*1e6
        q=(ppms<=ppm_cal).any(axis=1)
        qcal=np.array(list(set(fCalibrants[np.argmin(ppms[q],axis=1)])))
        qcal.sort()
        qmp=mp[q] 
        qmp=np.array([take_closest(qmp,i)  for i in qcal])
        pre_ppms=(qmp-qcal)/qcal*1e6

        # denoise calibrants
        dpp=pd.Series(pre_ppms,index=qcal)
        q1,q3=np.percentile(dpp,25),np.percentile(dpp,75) 
        fdpp=dpp[(dpp<q3+1.5*(q3-q1)) & (dpp>q3-1.5*(q3-q1))]
        peaks="start" 
        while len(peaks): 
            peaks, _=find_peaks(fdpp,prominence=(q3-q1)/2) #not sure why q3-q1 but it looked good for this dataset
            fdpp=fdpp[~fdpp.index.isin(fdpp.index[peaks])]
        qq=np.in1d(qcal,fdpp.index)
        qmp,pre_ppms,qcal=qmp[qq],pre_ppms[qq],qcal[qq]  
        
        
        #fitting
        pret=str(round(sum(abs(pre_ppms))/len(pre_ppms),1))
        y,x=qcal,m2c(qmp,sf,k0)  #true mass,#measured channel
        sf,k0 = least_squares(residual, [sf,k0], args=(x, y), method='lm',jac='2-point',max_nfev=3000).x
        qmp=c2m(tp,sf,k0)[q]
        qmp=np.array([take_closest(qmp,i)  for i in qcal])
        post_ppms=(qmp-qcal)/qcal*1e6
        postt=str(round(sum(abs(post_ppms))/len(post_ppms),1))
    
        #plotting
        fig,ax=plt.subplots()
        plt.scatter(qcal,pre_ppms,c=[(0.5, 0, 0, 0.3)])
        plt.scatter(qcal,post_ppms,c=[(0, 0.5, 0, 0.3)])
        plt.legend(["pre calibration","post calibration"])
        plt.xlabel("m/z")
        plt.ylabel("ppm mass error")
        plt.title("global_calibration")
        fig.savefig(itmfile.replace(Path(itmfile).suffix,"_tb"+str(bin_tof)+"_pb"+str(bin_pixels)+"_glob_cal_scat.png"),bbox_inches="tight",dpi=300)

        
        fig,ax=plt.subplots()
        y1, _, _ =plt.hist(pre_ppms,color=(0.5, 0, 0, 0.3))
        y2, _, _ =plt.hist(post_ppms,color=(0, 0.5, 0, 0.3))
        plt.vlines(np.mean(pre_ppms),0,np.hstack([y1,y2]).max(),color=(0.5, 0, 0, 1),linestyle='dashed')
        plt.vlines(np.mean(post_ppms),0,np.hstack([y1,y2]).max(),color=(0, 0.5, 0, 1),linestyle='dashed')
        plt.xlabel("ppm mass error")
        plt.ylabel("frequency")
        plt.legend(["pre: mean "+str(round(np.mean(pre_ppms),1))+ ", abs "+pret,
                    "post: mean "+str(round(np.mean(post_ppms),1))+ ", abs "+postt],
                    loc=[1.01,0])
        plt.title("global_calibration")
        fig.savefig(itmfile.replace(Path(itmfile).suffix,"_tb"+str(bin_tof)+"_pb"+str(bin_pixels)+"_glob_cal_hist.png"),bbox_inches="tight",dpi=300)

#%%
       
    ### Per scan analysis #### 
    ds=np.hstack([ds,ds[:,0].reshape(-1,1)]) #add bin_scan 
    if bin_scans: ds[:,5]=(ds[:,5]/bin_scans).astype(int)
    ds= ds[np.lexsort((ds[:,3],ds[:,2],ds[:,1],ds[:,5])) ]  #group binned pixels
    
    
    ### construct peak intervals 
    mp=c2m(p,sf,k0)
    calr=m2c(pd.DataFrame([[i,i*(1-ppm_cal/1e6),i*(1+ppm_cal/1e6)] for i in mp],
                          columns=["cal_channel","cal_lower","cal_upper"]),sf,k0)
    calr["interval"]=calr.apply(lambda x: np.arange(x["cal_lower"],x["cal_upper"]+1),axis=1)
    calr=calr.explode("interval")
    calr=calr[["cal_channel","interval"]].astype(int).set_index("interval")
    calr=calr.groupby(calr.index,sort=False).nth(0)
    tcalr=calr[calr.cal_channel.isin(tp)]
    
    pds=ds[np.in1d(ds[:,3].astype(int),tcalr.index.values.astype(int))]  # select only channels within top peak intervals
    dss=np.array_split(pds,np.argwhere(np.diff(pds[:,5])>0)[:,0]+1)      # split on scans
    if max_scans: dss=dss[:int(max_scans/bin_scans)]      
    
    if Calibrate_local:
        if Calibrate_global: fCalibrants=qcal #re-use same calibrants list
        else: fCalibrants=np.sort(np.array([pySPM.utils.get_mass(i) for i in Calibrate_local if i.endswith(mode_sign)])) 

    ### Per scan analysis
    totals=np.zeros(number_channels)
    ppm_map,rois,cdfs,kdfs=[],[],[],[]
    for d_ix,d in enumerate(dss):
        
        print(d_ix)
        sl=np.vstack([np.argwhere(np.diff(d[:,1])>0),np.argwhere(np.diff(d[:,2])>0)])[:,0]+1
        sl.sort()
        pixs=np.array_split(d[:,3],sl)
        
       
        #### Calibrate local ####
    
        if Calibrate_local:
            mppms,cmpixs=[],[]
            for ip,pix in enumerate(pixs):

                mpix=c2m(pix,sf,k0)
                closest=np.array([take_closest(mpix,i) for i in fCalibrants])
                ppms=(closest-fCalibrants)/ fCalibrants*1e6
                ppms=ppms[abs(ppms)<ppm_cal]
                
                b=ppms[np.argmin(abs(ppms))]
                if abs(b)>ppm_cal: b=np.nan
                mppms.append(b)
                
            #store 2d ppm map
            lcal=pd.DataFrame(d[np.hstack([0,sl])][:,1:3],columns=["x","y"])
            lcal["ppm"]=mppms
            pv=lcal.pivot(columns="x",index="y")
            pv=pv.fillna(0)
            pv.loc[:,:]=savgol2d(pv.values,window_size=5,order=3)
            pv.columns=pv.columns.droplevel()
            ppm_map.append(pv)
            
            #map back
            mpv=pv.melt()
            mpv.columns=["x","ppm"]
            mpv["y"]=pv.columns.astype(int).tolist()*len(pv.index)
            mpv["x"]=mpv["x"].astype(int)
            mpv["bin_scan"]=d_ix
            
            cdf=pd.DataFrame(d[:,1:3].astype(int),columns=["x","y"]).merge(mpv,on=["x","y"],how="left")
            cdfs.append(cdf.drop_duplicates())
            d[:,3]=m2c(c2m(d[:,3],sf,k0)*(1-cdf.ppm/1e6).values,sf,k0).astype(int)
            d=d[(d[:,3]<number_channels) & (d[:,3]>0)]   #error from ppm calibration 

        #### detect ROI ####

        #map peaks
        pks=d[np.in1d(d[:,3].astype(int),tcalr.index.astype(int)),1:4].astype(int)
        pks[:,2]=tcalr.loc[pks[:,2]].values[:,0]
        kdf=pd.DataFrame(pks,columns=["x","y","peak"]).groupby(["x","y","peak"]).size()
        kdf=kdf.reset_index().pivot(columns="peak",index=["x","y"]).fillna(0)
        kdf.columns = kdf.columns.droplevel()  
        kdfs.append(kdf)
        totals[kdf.columns]=totals[kdf.columns]+1 #add totals


    if Calibrate_local:
    
        cdfs=pd.concat(cdfs)
        cd=pd.DataFrame(ds[:,[1,2,5]].astype(int),columns=["x","y","bin_scan"]).merge(cdfs,on=["x","y","bin_scan"],how="left").fillna(0)
        ds[:,3]=m2c(c2m(ds[:,3],sf,k0)*(1-cd.ppm/1e6).values,sf,k0).astype(int)
        #ds[:,3]=m2c(c2m(ds[:,3],sf,k0)*(1+cd.ppm/1e6).values,sf,k0).astype(int)
        ds=ds[(ds[:,3]<number_channels) & (ds[:,3]>0)]   #error from ppm calibration 
        
        #plot summed 2d calibration map
        cal2d = reduce(lambda x, y: x.add(y, fill_value=0), ppm_map)/scans*bin_scans
        fig,ax=plt.subplots()
        sns.heatmap(cal2d)
        plt.title(Path(itmfile).stem)
        fig.savefig(itmfile.replace(Path(itmfile).suffix,"_tb"+str(bin_tof)+"_pb"+str(bin_pixels)+"_2dcal_map.png"),bbox_inches="tight",dpi=300)
    
    
        #%% Plot effects of local calibration
        
        if bool(Calibrate_local) & bool(Calibrate_local):
        
            u,uc=np.unique(ds[:,3],return_counts=True)#,return_inverse=True)
            Spectrum=nan_arr
            Spectrum[u.astype(int)]=uc
            Spectrum=pd.Series(Spectrum).interpolate().fillna(0).rolling(window=smoothing_window).mean()  #rolling mean will cause slight shifts in peak location.
            p,pi=find_peaks(Spectrum,prominence=prominence,distance=distance) 
            tp=p[np.sort(np.argsort(pi["prominences"])[-top_peaks:])]
            mp=c2m(tp,sf,k0)
    
            #get calibrants
            ppms=abs((qcal-mp.reshape(-1,1))/mp.reshape(-1,1))*1e6
            q=(ppms<=ppm_cal).any(axis=1)
            qmp=mp[q] 
            qmp=np.array([take_closest(qmp,i)  for i in qcal])
            
        
            post_local_ppms=(qmp-qcal)/qcal*1e6
            q=abs(post_local_ppms)<ppm_cal
            
            post_local_ppms=post_local_ppms[q]
            post_ppms=post_ppms[q]
            qcal=qcal[q]
            
            posttl=str(round(sum(abs(post_local_ppms))/len(post_local_ppms),1))
            postt=str(round(sum(abs(post_ppms))/len(post_ppms),1))
            
            #plotting
            fig,ax=plt.subplots()
            plt.scatter(qcal,post_ppms,c=(0.5, 0, 0, 0.3))
            plt.scatter(qcal,post_local_ppms,c=(0, 0.5, 0, 0.3))
            plt.legend(["pre local calibration","post local calibration"])
            plt.xlabel("m/z")
            plt.ylabel("ppm mass error")
            plt.title("local_calibration")
            fig.savefig(itmfile.replace(Path(itmfile).suffix,"_tb"+str(bin_tof)+"_pb"+str(bin_pixels)+"_loc_cal_scat.png"),bbox_inches="tight",dpi=300)

            fig,ax=plt.subplots()
            y1, _, _ =plt.hist(post_ppms,color=(0.5, 0, 0, 0.3))
            y2, _, _ =plt.hist(post_local_ppms,color=(0, 0.5, 0, 0.3))
            plt.vlines(np.mean(post_ppms),0,np.hstack([y1,y2]).max(),color=(0.5, 0, 0, 1),linestyle='dashed')
            plt.vlines(np.mean(post_local_ppms),0,np.hstack([y1,y2]).max(),color=(0, 0.5, 0, 1),linestyle='dashed')
            plt.xlabel("ppm mass error")
            plt.ylabel("frequency")
            plt.legend(["pre: mean "+str(round(np.mean(post_ppms),1))+ ", abs "+postt,
                        "post: mean "+str(round(np.mean(post_local_ppms),1))+ ", abs "+posttl],
                        loc=[1.01,0])
            plt.title("local_calibration")
            fig.savefig(itmfile.replace(Path(itmfile).suffix,"_tb"+str(bin_tof)+"_pb"+str(bin_pixels)+"_loc_cal_hist.png"),bbox_inches="tight",dpi=300)

  
        # #plot indifidual maps
        # for m in ppm_map:
        #     fig,ax=plt.subplots()
        #     sns.heatmap(m)
        
    #%% Scan total clustering on chemical signatures for ROI detection
    minsize=5
    max_channels=500 #otherwise kmeans can get memory overloaded
    allowed_channels=np.argwhere(totals>=minsize)[:,0]
    
    fkdfs=[]
    for i in kdfs:
        fkdf=i[i.columns[i.columns.isin(allowed_channels)]]
        if len(fkdf):
            fkdfs.append(fkdf)
    
    t=pd.concat(fkdfs).fillna(0) #concat is slow probably faster to flatten kdf and sum on flattened index
    ts=t.groupby(t.index).sum()
    ts[ts<minsize]=0
    ts=ts[ts.columns[(ts>0).any(axis=0)]]
    ts=ts[ts.columns.sort_values()]
    
    nts=ts.divide(ts.sum(axis=1).values,axis=0)
    
    if len(nts.columns)>max_channels:
        nts=nts[nts.sum().sort_values()[::-1][:max_channels].index.tolist()]

    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(ts.values)
    nts["roi"]=kmeans.labels_
    nts[["x","y"]]=nts.index.tolist()
    hm=nts[["x","y","roi"]].pivot(columns="x",index="y")#.fillna(0) #fill with fillna or with no_clusters+1
    hm.columns = hm.columns.droplevel()
    
    fig,ax=plt.subplots()
    sns.heatmap(hm)
    plt.title(Path(itmfile).stem)
    hm.to_csv(itmfile.replace(Path(itmfile).suffix,"_tb"+str(bin_tof)+"_pb"+str(bin_pixels)+"_ROI_map.tsv"),sep="\t")
    fig.savefig(itmfile.replace(Path(itmfile).suffix,"_tb"+str(bin_tof)+"_pb"+str(bin_pixels)+"_ROI_map.png"),bbox_inches="tight",dpi=300)
    #%% Get depth profiles


    print("Depth profile")
    pds=ds[np.in1d(ds[:,3].astype(int),calr.index.values.astype(int))]  # select only channels within peak intervals
    ddf=pd.DataFrame(pds,columns=["scan","x","y","tof","w","scan_bin"])
    ddf[["x","y"]]=ddf[["x","y"]].astype(int)
    ddf=ddf.set_index(["x","y"])
    
    dps=[]

    
    #sulphur plot
    fig,ax=plt.subplots()
    mp=c2m(np.arange(number_channels),sf,k0)
    ms=( mp>31.95) & ( mp<32.05)
    
    lines=[]
    for cl in range(n_clusters):
        
        xy=nts.loc[nts["roi"]==cl,["x","y"]].values.astype(int)
        roi=ddf.merge(pd.DataFrame(xy,columns=["x","y"]),on=["x","y"],how="inner")
        
        
        s=roi.groupby("tof")["w"].sum()
        Spectrum=np.zeros(number_channels)
        Spectrum[s.index.astype(int)]=s.values
        Spectrum=pd.Series(Spectrum).interpolate().fillna(0).rolling(window=smoothing_window).mean() 
        p,pi=find_peaks(Spectrum,prominence=prominence,distance=distance)
        
    
        # construct peak intervals (not 100% sure about this) 
        mp=c2m(p,sf,k0)
        calr=m2c(pd.DataFrame([[i,i*(1-ppm_cal/1e6),i*(1+ppm_cal/1e6)] for i in mp],
                              columns=["cal_channel","cal_lower","cal_upper"]),sf,k0)
        calr["interval"]=calr.apply(lambda x: np.arange(x["cal_lower"],x["cal_upper"]+1),axis=1)
        calr=calr.explode("interval")
        calr=calr[["cal_channel","interval"]].astype(int).set_index("interval")
        calr=calr.groupby(calr.index,sort=False).nth(0)
        roi=roi[np.in1d(roi.tof.astype(int).values,calr.index)]
        
    
        #sulphur plot
        line=ax.plot(c2m(np.arange(number_channels),sf,k0)[ms],Spectrum[ms],label="ROI "+str(cl))
        lines.append(line)
        
        
        roi=roi[["scan","x","y","tof","w"]].sort_values(by="scan")#.values 
        dp=roi.groupby(["scan","tof"])["w"].sum().reset_index() ##.pivot(columns="scan",index="tof")
        dp["tof"]=calr.loc[dp["tof"].astype(int)].values
        
        
        g=dp.groupby(["scan","tof"])["w"]
        
        Area=g.sum().reset_index().pivot(columns="scan",index="tof").fillna(0).astype(int)
        Area.columns=Area.columns.droplevel()
        Area.index=np.round(c2m(Area.index,sf,k0),3)
        Area.to_csv(itmfile.replace(Path(itmfile).suffix,"_tb"+str(bin_tof)+"_pb"+str(bin_pixels)+"_ROI"+str(cl)+"_depth_profile_Area.tsv"),sep="\t")

        Int=g.max().reset_index().pivot(columns="scan",index="tof").fillna(0).astype(int)
        Int.columns=Int.columns.droplevel()
        Int.index=np.round(c2m(Int.index,sf,k0),3)
        Int.to_csv(itmfile.replace(Path(itmfile).suffix,"_tb"+str(bin_tof)+"_pb"+str(bin_pixels)+"_ROI"+str(cl)+"_depth_profile_Int.tsv"),sep="\t")
        #find faster way than to csv, this is too slow

    plt.legend()
    plt.title(Path(itmfile).stem)
    fig.savefig(itmfile.replace(Path(itmfile).suffix,"_tb"+str(bin_tof)+"_pb"+str(bin_pixels)+"_sulphurplot.png"),bbox_inches="tight")


