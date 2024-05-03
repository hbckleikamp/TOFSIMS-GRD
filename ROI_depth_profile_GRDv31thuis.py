# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 11:38:30 2024

@author: hkleikamp
"""


#%% Modules

import numpy as np
import pandas as pd
from bisect import bisect_left #Fast pick close
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import least_squares
from scipy.signal import find_peaks, peak_widths
import pySPM


import seaborn as sns
from sklearn.cluster import KMeans
from functools import reduce
from sklearn import linear_model

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
cfile="E:/Wout_features/Sheath_calibrants.tsv"
csd=pd.read_csv(cfile,sep="\t")
#csd=csd[~csd["ToF-SIMS assignment"].str.contains("Au")] #remove gold
Calibrants=csd["ToF-SIMS assignment"].tolist() #extended calibrants (used for 3D ppm calibration)

emass=0.000548579909 #mass electron
elements=pd.read_csv("E:/Wout_features/utils/natural_isotope_abundances.tsv",sep="\t") #Nist isotopic abundances and monisiotopic masses
elements=elements[elements["Standard Isotope"]].set_index("symbol")["Relative Atomic Mass"]
elements=pd.concat([elements,pd.Series([-emass,emass],index=["+","-"])]) #add charges


#%%



# bin_tof=3 #3
# bin_pixels=3 #3 #bins in 2 directions: 2->, 3->9 data reduction
# bin_scans=3 #only used for ROI detection and local calibration
# ppm_cal=2000

# #peak picking
# smoothing_window=int(30/bin_tof) #window of rolling mean smoothing prior to peak picking
# max_width=2000 #on both sides (so actually *2)
# prominence=3 #minimum prominence
# distance=20  #minimum distance between peaks=
# top_peaks=100 #number of top peaks used for calibration and ROI detection


# Calibrate_global=True #exCalibrants #exCalibrants #Options: False #exCalibrants #Calibrants
# Calibrate_local=True #exCalibrants #False #only useful if samples have very inhomogenous heigt (single fibrils). exCalibrants #False #Calibrants #Calibrants #exCalibrants
# #calibrate Local is not working well at the moment?

# #Kmeans
# n_clusters=2 
# break_percentage=0.05 #stop analyzing scans after the amount of analyzed masses has not change by this amount (sample steady state)
# max_scans=False #50 #stop analyzing scans after this number (mostly for testing)

# min_count=3 #data reduction for spectra storage



# Delayed extraction needs higher mass ions
# Green, F. M., Ian S. Gilmore, and Martin P. Seah. "TOF-SIMS: Accurate mass scale calibration." Journal of the American Society for Mass Spectrometry 17 (2006): 514-523.
# Vanbellingen, Quentin P., et al. "Time‐of‐flight secondary ion mass spectrometry imaging of biological samples with delayed extraction for high mass and high spatial resolutions." Rapid Communications in Mass Spectrometry 29.13 (2015): 1187-1195.
### Olivier Scholder. (2018, November 28). scholi/pySPM: pySPM v0.2.16 (Version v0.2.16). Zenodo. http://doi.org/10.5281/zenodo.998575
#https://github.com/scholi/pySPM/blob/master/pySPM/ITM.py


#%% Functions

def m2c(m,sf,k0):    return np.round(  ( sf*np.sqrt(m) + k0 )  / bin_tof  ).astype(int)
def c2m(c,sf,k0):    return ((bin_tof * c - k0) / (sf)) ** 2   
def residual(p,x,y): return (y-c2m(x,*p))/y




def pick_peaks(ds,interp=True):
    gt=ds.groupby("tof",sort=False).size()
    Spectrum=nan_arr.copy()
    Spectrum[gt.index]=gt.values
    if interp:
        Spectrum=pd.Series(Spectrum).interpolate().fillna(0).rolling(window=smoothing_window).mean()  #rolling mean will cause slight shifts in peak location.
    p,pi=find_peaks(Spectrum,prominence=prominence,distance=distance) 
    tp=p[np.sort(np.argsort(pi["prominences"])[-top_peaks:])]
    return Spectrum,p,pi,tp
    

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
    return fftconvolve(Z, m, mode='valid')

def CalibrateGlobal(channels,calibrants,sf,k0,
                    
                    ppm_cal=2000,
                    min_calibrant_distance=0.5,
                    plot=True):


    try:
   
        plot=True       #test
        ppm_cal=2000    #test
        channels=tp     #test
        k0,sf=I.k0,I.sf #test
        min_calibrant_distance=0.5 #test
        
        mp=c2m(channels,sf,k0)
        
        #get calibrants
        ppms=abs((calibrants-mp.reshape(-1,1))/mp.reshape(-1,1))*1e6
        q=np.argwhere(ppms<=ppm_cal)
        qmp=mp[q[:,0]]
        qcal=calibrants[q[:,1]]
        ppm=(qmp-qcal)/qcal*1e6
    
        # denoise calibrants (maybe dtw is better?)
        dpp=pd.DataFrame(list(zip(qcal,qmp,channels[q[:,0]],ppm)),columns=["mass_q","mass_t","channel_t","ppm"])
        q1,q3=np.percentile(dpp.ppm,25),np.percentile(dpp.ppm,75) 
        fdpp=dpp[(dpp.ppm<q3+1.5*(q3-q1)) & (dpp.ppm>q3-1.5*(q3-q1))]
        peaks="start" 
        while len(peaks): 
            peaks, _=find_peaks(fdpp.ppm,prominence=(q3-q1)/2) 
            fdpp=fdpp[~fdpp.index.isin(fdpp.index[peaks])]
    
        #fitting #1
        y,x=fdpp.mass_q,fdpp.channel_t  #true mass,#measured channel
        sf,k0 = least_squares(residual, [sf,k0], args=(x, y), method='lm',jac='2-point',max_nfev=3000).x
        
        #filter fit
        post_ppms=(c2m(fdpp.channel_t,sf,k0)-y)/y*1e6
        fdpp["post_ppms"]=post_ppms
        q1,q3=np.percentile(post_ppms,25),np.percentile(post_ppms,75) 
        fdpp=fdpp[(post_ppms<q3+1.5*(q3-q1)) & (post_ppms>q3-1.5*(q3-q1))]
        
        #pick best calibrant 
        fdpp["g"]=np.where(np.diff(np.hstack([[0],fdpp.mass_q]))>=min_calibrant_distance,np.arange(len(fdpp)),np.nan)
        fdpp["g"]=fdpp["g"].ffill()
        fdpp["a"]=fdpp["post_ppms"].abs()
        fdpp=fdpp.sort_values(by=["g","a"]).groupby("g",sort=False).nth(0)
        
        #fitting #2
        y,x=fdpp.mass_q,fdpp.channel_t  #true mass,#measured channel
        sf,k0 = least_squares(residual, [sf,k0], args=(x, y), method='lm',jac='2-point',max_nfev=3000).x
        
        pre_ppms=fdpp.ppm
        pret=str(round(sum(abs(pre_ppms))/len(pre_ppms),1))
        post_ppms=(c2m(fdpp.channel_t,sf,k0)-y)/y*1e6
        postt=str(round(sum(abs(post_ppms))/len(post_ppms),1))
        calibrants=fdpp.mass_q.values #overwrite calibrants
        
        if plot:
    
            #plotting
            fig,ax=plt.subplots()
            plt.scatter(y,pre_ppms,c=[(0.5, 0, 0, 0.3)])
            plt.scatter(y,post_ppms,c=[(0, 0.5, 0, 0.3)])
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

    except:
        print("calibration failed!")

    return sf,k0,calibrants

def li(x,a):
    return a*x

def lin(x,a,b):
    return a*x+b

def monod(x,vmax,ks,b):
    return vmax*x/(ks+x)+b

def rmonod(x,vmax,ks,b):
    x=-x+x.max()
    return vmax*x/(ks+x)+b

# def comb2(guess):
#     a,b = guess
#     return (a*xqflt+b*yqflt-mqflt)**2



    
def r2(ydata,fitted_data):
    ss_res = np.sum((ydata- fitted_data)**2)
    ss_tot = np.sum((ydata-np.mean(ydata))**2)
    return  1 - (ss_res / ss_tot)


from scipy.optimize import curve_fit
def calibrate(data,method,p0="",reflect=False,Plot=False,fev=10000,bounds=(-np.inf, np.inf)):
    
    try:
    
        x,y=data[:,0],data[:,1]
        
        if reflect: #used for reverse monod
            x=-x+x.max()
            s=np.argsort(x)
            x,y=x[s],y[s]
                
        if method=="linear":
            if len(p0):
                popt, _ = curve_fit(lin,x,y,p0=p0,maxfev=fev,bounds=bounds)
            else:
                popt, _ = curve_fit(lin,x,y,maxfev=fev,bounds=bounds)
            f=lin(x,*popt)            
                
        if method=="monod":
            if len(p0):
                popt, _ = curve_fit(monod,x,y,p0=p0,maxfev=fev,bounds=bounds)
            else:
                popt, _ = curve_fit(monod,x,y,maxfev=fev,bounds=bounds)
            f=monod(x,*popt)
            

        ss_res = np.sum((y- f)**2)
        ss_tot = np.sum((y-np.mean(y))**2)
        rs= 1 - (ss_res / ss_tot)
        
        
        
        if Plot:
           fig,ax=plt.subplots()
           plt.scatter(x,y,s=1,label="PSMs")
           plt.plot(x,f,label=method+" fit r2: "+str(round(rs,3)),c="r")
           plt.ylabel("ppm mass shift")
           plt.xlabel("m/z")
           plt.title("global calibration")
           plt.legend(loc='best',markerscale=3)
           
           # if save: #extra toggle for testing
           #     fig.savefig(mzML_file.replace(".mzML","_global_fit.png"),dpi=300)
            
        return rs, popt 
    except Exception as error:
        print("An error occurred:", type(error).__name__, "–", error)
        return 0,[]


def kde_peak(data):
    try:
        x,y1=FFTKDE(bw="silverman").fit(data).evaluate(1000)
        return x[np.argmax(y1)]
    except:
        return np.nan
#%%
from KDEpy import FFTKDE
def CalibrateLocal(ds,p,pi):

    ""+1
    #%%
    pw=peak_widths(Spectrum, p, rel_height=0.5) #not really working?? (use right left bases instead?)
    pdf=pd.DataFrame(list(zip(p,pw[2].astype(int).tolist(), np.ceil(pw[3]).tolist())),columns=["peak","left","right"])
    pdf=pdf[pdf.peak.isin([m2c(take_closest(c2m(p,sf,k0),i),sf,k0) for i in calibrants])]
    pdf["range"]=pdf.apply(lambda x: list(range(int(x.left),int(x.right)+1)),axis=1)
    pdf["mass"]=c2m(pdf["peak"],sf,k0)
    
    e=pdf[["mass","range","peak"]].explode("range")
    e["ppm"]=(c2m(e.range,sf,k0)-e.mass)/e.mass*1e6
    e=e.set_index("range")
    eppm=nan_arr.copy(); ep=nan_arr.copy()     #view and copy 
    eppm[e.index.astype(int)]=e.ppm            #converts tof to ppm
    ep[e.index.astype(int)]=e.peak.astype(int) #converts tof to peak 
    
    cds=ds.loc[ds["tof"].isin(e.index),["x","y","tof"]].values
    cds=cds[abs(eppm[cds[:,2]])<200] #make into parameter
    cds= cds[np.lexsort((cds[:,2],cds[:,0],cds[:,1])) ] 
    cds=np.hstack([cds,ep[cds[:,2]].reshape(-1,1)])
    

    dss=np.array_split(cds,np.argwhere((np.diff(cds[:,:2],axis=0)>0).sum(axis=1))[:,0]+1) #split on xy coords

    from numpy.linalg import lstsq as np_lstsq

        
    kds=[]

    for d in dss:
        
        #fast peak picking https://stackoverflow.com/questions/22124332/group-argmax-argmin-over-partitioning-indices-in-numpy
        unique,counts=np.unique(d[:,2:],axis=0,return_counts=True)
        split_at=np.argwhere(np.diff(unique[:,1]))[:,0]+1     

        group_lengths = np.diff(np.hstack([0, split_at, len(counts)]))
        n_groups = len(group_lengths)
        index = np.repeat(np.arange(n_groups), group_lengths)
        maxima = np.maximum.reduceat(counts, np.hstack([0, split_at]))
        all_argmax = np.flatnonzero(np.repeat(maxima, group_lengths) == counts)
        result = np.empty(len(group_lengths), dtype='i')
        result[index[all_argmax[::-1]]] = all_argmax[::-1]
        
        x,y=unique[result,0].reshape(-1,1),unique[result,1].reshape(-1,1)
        
     
        a,b,_,_=np_lstsq(x, y)
        r2=1-(np.sum((x-(y*a+b))**2)/np.sum((x-np.mean(x))**2))
        
        # ppm=(c2m(y,sf,k0)-c2m(x,sf,k0))/c2m(y,sf,k0)*1e6
        # kds.append(list(d[0,:2])+[a[0][0],b[0],r2,len(x),np.median(ppm)])
        kds.append(list(d[0,:2])+[a[0][0],b[0],r2,len(x),np.median(y-x)])

    d=pd.DataFrame(kds,columns=["x","y","a","b","r2","n","delta_channels"])
    
    mpv=d.pivot(index="y",columns="x",values="delta_channels")
                

    fig,ax=plt.subplots()
    sns.heatmap(mpv,robust=True)
    plt.title("median delta tof")


    rpv=d.pivot(index="y",columns="x",values="r2")

   
    ##### Extract non-linear TOF correction
    
    #pick top pixels
    cutoff=75
    q=rpv>pd.Series(rpv.values.flatten()).quantile(cutoff/100)
    flat_pixels=d[d.r2>d.r2.quantile(cutoff/100)]
  
    #initial linear model
    ppix=np.array([[0,0],[xpix,0],[0,ypix]])
    lmod=linear_model.LinearRegression().fit(flat_pixels[["x","y"]], flat_pixels["delta_channels"])
    lm=lmod.predict(ppix)
    d["lmfittd"]=lmod.predict(d[["x","y"]])
    
    fpv=d.pivot(index="y",columns="x",values="lmfittd")
    fig,ax=plt.subplots()
    sns.heatmap(fpv,robust=True)
    plt.title("linear model")

    #Primary non-linear fit optimization
    xp0=lmod.predict(np.vstack([np.zeros(ypix),np.arange(ypix)]).T)
    xpe=lmod.predict(np.vstack([np.ones(ypix)*ypix,np.arange(ypix)]).T)  
    xsc=((mpv-xp0)/(xpe-xp0))[q].unstack().dropna().reset_index()
        
    vx=xsc[["x",0]].values
    lrsx,lpoptx=calibrate(vx,method="linear")
    p0x=[lin(vx[-1,0],*lpoptx),vx[-1,0]/2,0]
    mrsx,mpoptx=calibrate(vx,method="monod",p0=p0x)
    rmrsx,rmpoptx=calibrate(vx,method="monod",p0=p0x[::-1],reflect=True)
    
    yp0=lmod.predict(np.vstack([np.arange(xpix),np.zeros(xpix)]).T)
    ype=lmod.predict(np.vstack([np.arange(xpix),np.ones(xpix)*xpix]).T)
    ysc=((mpv-yp0)/(ype-yp0))[q].unstack().dropna().reset_index()
      
    vy=ysc[["y",0]].values
    lrsy,lpopty=calibrate(vy,method="linear")
    p0y=[lin(vy[-1,0],*lpopty),vy[-1,0]/2,0]
    mrsy,mpopty=calibrate(vy,method="monod",p0=p0y)                      #monod
    rmrsy,rmpopty=calibrate(vy,method="monod",p0=p0y[::-1],reflect=True) #reverse monod
    
    #Secondary non-linear fit optimization
    bfit=pd.Series([lrsx,mrsx,rmrsx,lrsy,mrsy,rmrsy],index=["linear_X","monod_X","rmonod_X","linear_Y","monod_Y","rmonod_Y"]).idxmax()
    
    
    
    if bfit.endswith("X"): #first fit x
    
        bfitx=bfit
        if bfit.startswith("monod"):   xh=monod(np.arange(xpix),*mpoptx)*(xpe-xp0).reshape(-1,1)+xp0
        if bfit.startswith("lin"):     xh=lin(np.arange(xpix),*lpoptx)*(xpe-xp0).reshape(-1,1)+xp0   
        if bfit.startswith("rmonod"):  xh=(monod(np.arange(xpix),*rmpoptx)*(xpe-xp0).reshape(-1,1)+xp0)[::-1]             
            
        xshift=pd.DataFrame(xh,columns=np.arange(xpix),index=np.arange(ypix))
        rm=(mpv-xshift)[q]
        ms=abs(rm).max(axis=1).values
        vy=(rm/ms).median(axis=1).reset_index().values
        
        lrsy,lpopty=calibrate(vy,method="linear")
        p0y=[lin(vy[-1,0],*lpopty),vy[-1,0]/2,0]
        mrsy,mpopty=calibrate(vy,method="monod",p0=p0y)
        rmrsy,rmpopty=calibrate(vy,method="monod",p0=p0y[::-1],reflect=True)
        
        bfity=pd.Series([lrsy,mrsy,rmrsy],index=["linear_Y","monod_Y","rmonod_Y"]).idxmax()
        if bfity.startswith("monod"):   yh=monod(np.arange(ypix),*mpopty).reshape(-1,1)*np.ones([1,ypix])
        if bfity.startswith("lin"):     yh=lin(np.arange(ypix),*lpopty).reshape(-1,1)*np.ones([1,ypix])
        if bfity.startswith("rmonod"):  yh=(monod(np.arange(ypix),*rmpopty).reshape(-1,1)*np.ones([1,ypix]))[::-1]
        
        yshift=pd.DataFrame(yh,columns=np.arange(xpix),index=np.arange(ypix))
        yr,_=np.polydiv(rm[q].unstack().dropna().values,yshift[q].unstack().dropna().values)
        yshift=yshift*yr[0]
    
        #correct popt
        lpopty[-1]*=yr[0];   lpopty[0]*=yr[0]
        mpopty[-1]*=yr[0];   mpopty[0]*=yr[0]
        rmpopty[-1]*=yr[0]; rmpopty[0]*=yr[0]
    
    
    if bfit.endswith("Y"): 
            
        bfity=bfit
        if bfit.startswith("monod"):  yh=monod(np.arange(ypix),*mpopty)*(ype-yp0).reshape(-1,1)+yp0
        if bfit.startswith("lin"):    yh=lin(np.arange(ypix),*lpopty)*(ype-yp0).reshape(-1,1)+yp0   
        if bfit.startswith("rmonod"):  yh=(monod(np.arange(ypix),*rmpopty)*(ype-yp0).reshape(-1,1)+yp0)[::-1]             
            
        yshift=pd.DataFrame(yh,columns=np.arange(xpix),index=np.arange(ypix))
        rm=(mpv-yshift)[q]
        ms=abs(rm).max(axis=0).values
        vx=(rm/ms).median(axis=0).reset_index().values
        
        lrsx,lpoptx=calibrate(vx,method="linear")
        p0x=[lin(vx[-1,0],*lpopty),vx[-1,0]/2,0]
        mrsx,mpoptx=calibrate(vx,method="monod",p0=p0x)
        rmrsx,rmpoptx=calibrate(vx,method="monod",p0=p0x[::-1],reflect=True)
        
        bfitx=pd.Series([lrsx,mrsx,rmrsx],index=["linear_X","monod_X","rmonod_X"]).idxmax()
        if bfitx.startswith("monod"):   xh=monod(np.arange(xpix),*mpoptx)*np.ones([xpix,1])
        if bfitx.startswith("lin"):     xh=lin(np.arange(xpix),*lpoptx)*np.ones([xpix,1])
        if bfitx.startswith("rmonod"):  xh=np.fliplr(monod(np.arange(xpix),*rmpoptx)*np.ones([xpix,1]))
        
        xshift=pd.DataFrame(xh,columns=np.arange(xpix),index=np.arange(ypix))
        xr,_=np.polydiv(rm[q].unstack().dropna().values,xshift[q].unstack().dropna().values)
        xshift=xshift*xr[0]
        
        # #correct popt
        # lpoptx[-1]*=xr[0];   lpoptx[0]*=xr[0]
        # mpoptx[-1]*=xr[0];   mpoptx[0]*=xr[0]
        # rmpoptx[-1]*=xr[0]; rmpoptx[0]*=xr[0]
        
    ## Linear combination of non-linear fits
    xqflt=xshift[q].unstack().dropna().values
    yqflt=yshift[q].unstack().dropna().values
    mqflt=mpv[q].unstack().dropna().values
    
    def comb2(guess):
        a,b = guess
        return (a*xqflt+b*yqflt-mqflt)**2
    
    results = least_squares(comb2, [1,1])
    xshift=xshift*results.x[0]
    yshift=yshift*results.x[1]
    
    fig,ax=plt.subplots()
    sns.heatmap(yshift+xshift,robust=True)
    plt.title("non-linear model")
    plt.xlabel("y"); plt.ylabel("y")
    
    
    #refit again!
    
    ""+1
    #correct popt
    lpoptx[-1]*=results.x[0];   lpoptx[0]*=results.x[0]
    mpoptx[-1]*=results.x[0];   mpoptx[0]*=results.x[0]
    rmpoptx[-1]*=results.x[0]; rmpoptx[0]*=results.x[0]
    lpopty[-1]*=results.x[1];   lpopty[0]*=results.x[1]
    mpopty[-1]*=results.x[1];   mpopty[0]*=results.x[1]
    rmpopty[-1]*=results.x[1]; rmpopty[0]*=results.x[1]
    

    
    
    # fig,ax=plt.subplots()
    # sns.heatmap(mpv-xshift,robust=True)
    # plt.title("xshift only")
    
        
    # fig,ax=plt.subplots()
    # sns.heatmap(mpv-xshift*results.x[0]-yshift*results.x[1],robust=True)
    # plt.title("xshift+yshift")
    
    
    # fig,ax=plt.subplots()
    # sns.heatmap(mpv-fpv,robust=True)
    # plt.title("linear model")
    
    
    # rs=[
    # np.mean(abs(mpv[q]-fpv[q]).unstack().dropna().values),
    # np.mean(abs(mpv[q]-xshift[q]).unstack().dropna().values),
    # np.mean(abs(mpv[q]-xshift[q]*results.x[0]-yshift[q]*results.x[1]).unstack().dropna().values)]
    
    ""+1
    
    #%%
    
    
    
    
    
    #per scan analysis

    #here you need an extended number of calibrants
    #pw=peak_widths(Spectrum, p, rel_height=0.5) #not really working?? (use right left bases instead?)
    pdf=pd.DataFrame(list(zip(p,pw[2].astype(int).tolist(), np.ceil(pw[3]).tolist())),columns=["peak","left","right"])
    pdf["range"]=pdf.apply(lambda x: list(range(int(x.left),int(x.right)+1)),axis=1)
    pdf["mass"]=c2m(pdf["peak"],sf,k0)
    
    e=pdf[["mass","range","peak"]].explode("range")
    e["ppm"]=(c2m(e.range,sf,k0)-e.mass)/e.mass*1e6
    e=e.set_index("range")
    
    
    eppm=nan_arr.copy(); ep=nan_arr.copy()     #view and copy 
    eppm[e.index.astype(int)]=e.ppm            #converts tof to ppm
    ep[e.index.astype(int)]=e.peak.astype(int) #converts tof to peak 
     
    
    ### Assess image drift
    
    #flat_pixels
    gds=q.unstack()[q.unstack().values].reset_index()[["x","y"]].merge(ds,how="left",on=["x","y"])
    cds=gds.loc[gds["tof"].isin(e.index),["scan","x","y","tof"]].values.astype(int)
    cds=cds[abs(eppm[cds[:,3]])<200] #make into parameter
    cds=np.hstack([cds,ep[cds[:,3]].reshape(-1,1)])
    cds= cds[np.lexsort((cds[:,2],cds[:,1],cds[:,0])) ] 
    dss=np.array_split(cds,np.argwhere(np.diff(cds[:,0],axis=0)>0)[:,0]+1) #split on xy coords

    lmls=[]
    #%%
    scan_fits=[]
    for ix,sd in enumerate(dss):
        print(ix)
        sds=np.array_split(sd,np.argwhere((np.diff(sd[:,1:3],axis=0)>0).sum(axis=1))[:,0]+1) #split on xy coords
        
        
        # kds=[np.hstack([d[0,1:3],np.median(d[:,4]-d[:,3])])  for d in sds]
        # df=pd.DataFrame(np.vstack(kds),columns=["x","y","delta_channels"])


        kds=np.vstack([np.hstack([d[0,1:3],np.median(d[:,4]-d[:,3])])  for d in sds])
        x,y,ld=kds[:,0],kds[:,1],kds[:,2]
                
        def c_xl_yl(guess):
            ax,bx,  ay,by  =guess
            return (ld  -ax*x-bx  -ay*y-by )**2
        
        def c_xl_ym(guess):
            ax,bx,  vmax,ks,by =guess
            return (ld  -ax*x-bx  -vmax*y/(ks+y)-by   )**2
        
        def c_xl_ymr(guess):
            ax,bx,  vmax,ks,by =guess
            ry=-y+y.max()
            return (ld  -ax*x-bx  -vmax*y/(ks+ry)-by   )**2
        
        def c_xm_yl(guess):
            vmax,ks,bx, ay,by=guess
            return (ld  -vmax*x/(ks+x)-bx  -ay*y-by     )**2
        
        def c_xm_ym(guess):
            xvmax,xks,bx,  yvmax,yks,by=guess
            return (ld  -xvmax*x/(xks+x)-bx  -yvmax*y/(yks+y)-by   )**2
        
        def c_xm_ymr(guess):
            xvmax,xks,bx,  yvmax,yks,by=guess
            ry=-y+y.max()
            return (ld  -xvmax*x/(xks+x)-bx  -yvmax*ry/(yks+ry)-by   )**2
        
        def c_xmr_yl(guess):
            xvmax,xks,bx,   ay,by  =guess
            rx=-x+x.max()
            return (ld  -xvmax*rx/(xks+rx)-bx  -ay*y-by )**2
        
        def c_xmr_ym(guess):
            xvmax,xks,bx,  yvmax,yks,by=guess
            rx=-x+x.max()
            return (ld  -xvmax*rx/(xks+rx)-bx  -yvmax*y/(yks+y)-by   )**2
        
        def c_xmr_ymr(guess):
            xvmax,xks,bx,  yvmax,yks,by=guess
            rx=-x+x.max()
            ry=-y+y.max()
            return (ld  -xvmax*rx/(xks+rx)-bx  -yvmax*ry/(yks+ry)-by   )**2

        # Determine fitting method
        
        if bfitx.startswith("lin"):
            if bfity.startswith("lin"):    scan_fits.append(pd.DataFrame([least_squares(c_xl_yl,  lpoptx.tolist()+ lpopty.tolist()).x],columns=["ax","bx",  "ay","by"]))
            if bfity.startswith("monod"):  scan_fits.append(pd.DataFrame([least_squares(c_xl_ym,  lpoptx.tolist()+ mpopty.tolist()).x],columns=["ax","bx",  "yvmax","yks","by"]))
            if bfity.startswith("rmonod"): scan_fits.append(pd.DataFrame([least_squares(c_xl_ymr, lpoptx.tolist()+rmpopty.tolist()).x],columns=["ax","bx",  "yvmax","yks","by"]))
        
        if bfitx.startswith("monod"):
            if bfity.startswith("lin"):    scan_fits.append(pd.DataFrame([least_squares(c_xm_yl,  mpoptx.tolist()+ lpopty.tolist()).x],columns=["xvmax","xks","bx",  "ay","by"]))
            if bfity.startswith("monod"):  scan_fits.append(pd.DataFrame([least_squares(c_xm_ym,  mpoptx.tolist()+ mpopty.tolist()).x],columns=["xvmax","xks","bx",  "yvmax","yks","by"]))
            if bfity.startswith("rmonod"): scan_fits.append(pd.DataFrame([least_squares(c_xm_ymr, mpoptx.tolist()+rmpopty.tolist()).x],columns=["xvmax","xks","bx",  "yvmax","yks","by"]))
            
        if bfitx.startswith("rmonod"):
            if bfity.startswith("lin"):    scan_fits.append(pd.DataFrame([least_squares(c_xmr_yl,  rmpoptx.tolist()+ lpopty.tolist()).x],columns=["xvmax","xks","bx",  "ay","by"]))
            if bfity.startswith("monod"):  scan_fits.append(pd.DataFrame([least_squares(c_xmr_ym,  rmpoptx.tolist()+ mpopty.tolist()).x],columns=["xvmax","xks","bx",  "yvmax","yks","by"]))
            if bfity.startswith("rmonod"): scan_fits.append(pd.DataFrame([least_squares(c_xmr_ymr, rmpoptx.tolist()+rmpopty.tolist()).x],columns=["xvmax","xks","bx",  "yvmax","yks","by"]))
        
        #fit
        
        #correct ppm
    
    scan_fits=pd.concat(scan_fits)
    
        #%%
        r=pd.pivot(df,columns="x",index="y",values="delta_channels")


        ab=[]
        for n,v in r.iterrows():
            v=v.dropna()
            if len(v)>2:
                x,y=v.index.values,v.values
                
                fig,ax=plt.subplots()
                plt.scatter(x,y)
                plt.title("x"+str(n))
                
                a,b,_,_=np_lstsq(x.reshape(-1,1), y.reshape(-1,1))
                ab.append(np.hstack([a[0],b]))
        ab=np.vstack(ab)
        xcor,xres=np.median(ab,axis=0)


        ab=[]
        for n,v in r.T.iterrows():
            v=v.dropna()
            if len(v)>2:
                x,y=v.index.values,v.values
                        
                fig,ax=plt.subplots()
                plt.scatter(x,y)
                plt.title("y"+str(n))
                
                a,b,_,_=np_lstsq(x.reshape(-1,1), y.reshape(-1,1))
                ab.append(np.hstack([a[0],b]))
        ab=np.vstack(ab)
        ycor,yres=np.median(ab,axis=0)
        
        
        
        #%%
        
        ""+1
        
        # if ix<30:
        #     fig,ax=plt.subplots()
        #     sns.heatmap(df.pivot(columns="x",index="y",values="delta_channels"),robust=True,vmax=10,vmin=-10)
        #     plt.title(ix)
        # else:
        #     break
        
        lmls.append(np.hstack([[ix],linear_model.LinearRegression().fit(df[["x","y"]], df["delta_channels"]).predict(ppix)]))
    
    lmls=pd.DataFrame(np.vstack(lmls),columns=["scan","x0y0","xend","yend"])

#%%

# #%% Image drift correction with trilateration (save for later)

# import scipy
# from scipy.optimize import least_squares

# # Test point is 2000, 2000, -100, 0

# x1, y1,  dist_1 = (   0,   0, lm[0])
# x2, y2,  dist_2 = (   128, 0, lm[1])
# x3, y3,  dist_3 = ( 0,   128, lm[2])

# x1, y1,  dist_1 = (   0,   0, mpv.loc[124,0])
# x2, y2,  dist_2 = (   64, 0, mpv.loc[124,64])
# x3, y3,  dist_3 = ( 128,   0, mpv.loc[124,127])

# #Define a function that evaluates the equations
# def trilat( guess ):
#     x, y, r, sc = guess
#     return (
#         (x - x1)**2 + (y - y1)**2 - (dist_1*sc - r )**2,
#         (x - x2)**2 + (y - y2)**2 - (dist_2*sc - r )**2,
#         (x - x3)**2 + (y - y3)**2 - (dist_3*sc - r )**2,
#     )

# xcor,ycor=(lm[1]-lm[0])/xpix,(lm[2]-lm[0])/ypix
# x0=xpix/2
# dx=-lm[0]/xcor
# initial_guess = (0, 5,  5,20)
# results = least_squares(trilat, initial_guess)

# #add bounds
#     	#less then dx
#         #negative 

# #update inital guess:
#     #dy: count from the middle
#     #sx: pythagoras

# #then you need to scale 
# fig,ax=plt.subplots()
# plt.scatter(x1,y1)
# plt.scatter(x2,y2)
# plt.scatter(x3,y3)
# plt.scatter(results.x[0],results.x[1])

# sc=1.8#results.x[-1]

# circle1=plt.Circle((x1,y1),dist_1*sc, fill=False)
# circle2=plt.Circle((x2,y2),dist_2*sc, fill=False)
# circle3=plt.Circle((x3,y3),dist_3*sc, fill=False)

# plt.title("initial trialateration")
# plt.legend([str(i)[1:-1] for i in ppix],loc='upper right')
# ax.add_patch(circle1)
# ax.add_patch(circle2)
# ax.add_patch(circle3)
# plt.title(Path(itmfile).stem)
# # ""+1




#%%



#%%
itmfiles=["E:/Antwerp/ToF-SIMS Louvain/20240117_cablebacteria_backup/N_Slide I spot VIII.itm",
"E:/Antwerp/ToF-SIMS Louvain/20240117_cablebacteria_backup/N_Slide I spot IV.itm",
"E:/Antwerp/ToF-SIMS Louvain/20240117_cablebacteria_backup/N_Slide I spot VIIa.itm",
"E:/Antwerp/ToF-SIMS Louvain/20240117_cablebacteria_backup/N_Slide I spot VIIb.itm"]

itafiles=["E:/Antwerp/ToF-SIMS Louvain/20240117_cablebacteria_backup/N_Slide I spot VIII_0.ita",
"E:/Antwerp/ToF-SIMS Louvain/20240117_cablebacteria_backup/N_Slide I spot IV_0.ita",
"E:/Antwerp/ToF-SIMS Louvain/20240117_cablebacteria_backup/N_Slide I spot VIIa_0.ita",
"E:/Antwerp/ToF-SIMS Louvain/20240117_cablebacteria_backup/N_Slide I spot VIIb_0.ita"]

itmfiles=["E:/Antwerp/ToF-SIMS Louvain/20240117_cablebacteria_backup/N_Slide I spot VIII.itm"]

itafiles=["E:/Antwerp/ToF-SIMS Louvain/20240117_cablebacteria_backup/N_Slide I spot VIII_0.ita"]


# itmfiles=["E:/Antwerp/ToF-SIMS Louvain/20240131_CableBacteria/P_Oregon.itm"]
# itafiles=["E:/Antwerp/ToF-SIMS Louvain/20240131_CableBacteria/P_Oregon_0.ita"]

bin_tof=1 #3
bin_pixels=1 #3 #bins in 2 directions: 2->, 3->9 data reduction
bin_scans=10 #only used for ROI detection and local calibration
top_peaks=200

#peak picking
smoothing_window=int(30/bin_tof) #window of rolling mean smoothing prior to peak picking
max_width=2000 #on both sides (so actually *2)
prominence=3 #minimum prominence
distance=20  #minimum distance between peaks=
top_peaks=100 #number of top peaks used for calibration and ROI detection
Calibrate_global=True #exCalibrants #exCalibrants #Options: False #exCalibrants #Calibrants
Calibrate_local=True #exCalibrants #False #only useful if samples have very inhomogenous heigt (single fibrils). exCalibrants #False #Calibrants #Calibrants #exCalibrants


itmfiles.sort()
itafiles.sort()


grd_exe="C:/Program Files (x86)/ION-TOF/SurfaceLab 6/bin/ITRawExport.exe"

import subprocess
import os
for ixf,itmfile in enumerate(itmfiles):
    
    grdfile=itmfile+".grd"
    if not os.path.exists(grdfile):
        command='"'+grd_exe+'"'+' "'+itmfile+'"'
        print(command)
        stdout, stderr =subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    
    I = pySPM.ITM(itmfile)
    xpix,ypix= I.size["pixels"]["x"], I.size["pixels"]["y"]
    
    
    A = pySPM.ITA(itafiles[ixf])
#    ""+1
    #%% 
   
    
    
    # ### Get ITA metadata ###
    # positions={i:A.get_value("Instrument.Stage.Position."+i)["float"]*1e6 for i in ["R","T","U","V","W","X","Y","Z"]}
    # meta_S=A.root.goto('Meta/Video Snapshot').dict_list()
    # meta_I=A.root.goto('Meta/SI Image').dict_list()
    # sx,  sy  =meta_S['res_x']['ulong']              ,meta_S['res_y']['ulong']               #pixels
    # x_um,y_um=meta_S["fieldofview_x"]["float"]*10**6,meta_S["fieldofview_y"]["float"]*10**6 #real FOV
    # zoom_factor=meta_I["zoomfactor"]["float"]
    # scale_x, scale_y = x_um/sx, y_um/sy  #pixels to len
    # xlen=A.size["real"]["x"]*1e6/zoom_factor/scale_x/xpix #um per pixel
    # ylen=A.size["real"]["y"]*1e6/zoom_factor/scale_y/ypix #um per pixel
    
    
    # (positions["V"]-positions["Y"])**2+(positions["U"]-positions["X"])**2
    
    # plt.scatter(positions["X"],positions["Y"])
    # plt.scatter(positions["U"],positions["V"])
    
    # 588	Registration.Raster.CenterPosition.R	0.00028246484 
    # 589	Registration.Raster.CenterPosition.T	-0.00019700205 
    # 590	Registration.Raster.CenterPosition.X	-21.255709 mm
    # 591	Registration.Raster.CenterPosition.Y	-22.514408 mm
    # 592	Registration.Raster.CenterPosition.Z	-0.038621586 mm
    # 608	Registration.Raster.TotalFieldOfView.X	0.15 mm
    # 609	Registration.Raster.TotalFieldOfView.Y	0.15 mm
    
    
    # 352	Instrument.Stage.Position.R	0.0162 deg
    # 353	Instrument.Stage.Position.T	-0.0113 deg
    # 354	Instrument.Stage.Position.U	-21.2621 mm
    # 355	Instrument.Stage.Position.V	-22.5083 mm
    # 356	Instrument.Stage.Position.W	25.0386 mm
    
    
    # 357	Instrument.Stage.Position.X	-21.2557 mm
    # 358	Instrument.Stage.Position.Y	-22.5144 mm
    # 359	Instrument.Stage.Position.Z	-0.0386 mm
    
    #%%
    
    ### Get ITM metadata ###
    xpix,ypix= I.size["pixels"]["x"], I.size["pixels"]["y"]
    k0,sf=I.k0,I.sf
    mode_sign={"positive":"+","negative":"-"}.get((I.polarity).lower())
    calibrants=np.sort(np.array([pySPM.utils.get_mass(i) for i in Calibrants if i.endswith(mode_sign)])) #shortended calibrants list
    scans=I.root.goto("propend/Measurement.ScanNumber").get_key_value()["int"]
    number_channels = int(round(I.root.goto('propend/Measurement.CycleTime').get_key_value()['float'] \
                                / I.root.goto('propend/Registration.TimeResolution').get_key_value()['float'])/bin_tof)
    nan_arr=np.array([np.nan]*number_channels)    

    #vals=pd.Series(I.getValues()).explode().str.replace("\x00"," ").reset_index().astype(str).to_clipboard()
    #vals=pd.Series(A.getValues()).explode().to_clipboard()

    ### Unpack raw scans ###
    with open(grdfile,"rb") as fin:
        ds=pd.DataFrame(np.fromfile(fin,dtype=np.uint32).reshape(-1,5)[:,[0,2,3,4]],
                        columns=["scan","x","y","tof"]).astype(np.uint32)

    
    ### Binning ###
    ds[["x","y"]]=(ds[["x","y"]]/bin_pixels).astype(int)
    ds["tof"]=  (ds["tof"] /bin_tof   ).astype(int) 
    ds=ds.astype(np.uint32)

    ### calculate peaks ###

    Calibration_rounds=5
    
    #first round
    Spectrum,p,pi,tp=pick_peaks(ds)
    sf,k0,callibrants=CalibrateGlobal(tp,calibrants,sf,k0)
    xs=c2m(np.arange(number_channels),sf,k0)
    
    #2nd round
    ds=CalibrateLocal(ds,p,pi)
    # Spectrum2,p2,pi,tp=pick_peaks(ds)
    # sf,k0,_=CalibrateGlobal(tp,calibrants,sf,k0)
    # xs2=c2m(np.arange(number_channels),sf,k0)
    
#     #3nd round
#     ds=CalibrateLocal(ds,p,pi)
#     Spectrum2,p2,pi,tp=pick_peaks(ds)
#     sf,k0,_=CalibrateGlobal(tp,calibrants,sf,k0)
#     xs2=c2m(np.arange(number_channels),sf,k0)
    
#     #4nd round
#     ds=CalibrateLocal(ds,p,pi)
#     Spectrum2,p2,pi,tp=pick_peaks(ds)
#     sf,k0,_=CalibrateGlobal(tp,calibrants,sf,k0)
#     xs2=c2m(np.arange(number_channels),sf,k0)
    
#     #5nd round
#     ds=CalibrateLocal(ds,p,pi)
#     Spectrum2,p2,pi,tp=pick_peaks(ds)
#     sf,k0,_=CalibrateGlobal(tp,calibrants,sf,k0)
#     xs2=c2m(np.arange(number_channels),sf,k0)
    
#     #6th round
#     ds=CalibrateLocal(ds,p,pi)
#     Spectrum2,p2,pi,tp=pick_peaks(ds)
#     sf,k0,_=CalibrateGlobal(tp,calibrants,sf,k0)
#     xs2=c2m(np.arange(number_channels),sf,k0)
    
    
#     # #%%
#     # fig,ax=plt.subplots()
#     # plt.plot(c2m(np.arange(number_channels),sf,k0),Spectrum/Spectrum.max(),alpha=0.5)
#     # plt.plot(c2m(np.arange(number_channels),sf,k0),Spectrum2/Spectrum2.max(),alpha=0.5)
    
    
#     # plt.xlim(11.98,12.02)
#     # #plt.ylim(0,13000)
   
#     # #%%
    
#     # fig,ax=plt.subplots()
#     # plt.plot(c2m(np.arange(number_channels),sf,k0),Spectrum,alpha=0.5)
#     # #plt.plot(c2m(np.arange(number_channels),sf,k0),Spectrum2,alpha=0.5)
    
    
#     # plt.xlim(10,20)
#     # plt.ylim(0,1000)
   
    
#     #%%
#     fig,ax=plt.subplots()
#     plt.plot(xs,Spectrum-Spectrum2,alpha=0.5)
#     #plt.xlim(10,20)
#     plt.xlim(11.98,12.02)
#     #%%
#     fig,ax=plt.subplots()
#     plt.plot(xs,Spectrum/Spectrum.max(),alpha=0.5)
#     plt.plot(xs2,Spectrum2/Spectrum2.max(),alpha=0.5)
#     plt.vlines(12.000549,0,0.2)
#     plt.xlim(11.98,12.02)
    
    
#     #%%
#     fig,ax=plt.subplots()
#     plt.plot(xs,Spectrum,alpha=0.5)
#     plt.plot(xs2,Spectrum2,alpha=0.5)

#     plt.xlim(20,30)
    
#     #plt.ylim(0,13000)
#     #%%
    
#     # for xr in range(Calibration_rounds):
       
#     #     if xr:
#     #         Spectrum,p,pi,tp=pick_peaks(ds)
#     #         if Calibrate_global: sf,k0,_=CalibrateGlobal(tp,calibrants,sf,k0)
        
#     #     if Calibrate_local :               ds=CalibrateLocal(ds,p,pi)
    




    
#     #%% Calibrate Local
    

    
#     #Do this per scan because of lateral shifts???
    
#     #Detect ROI (is this needed for NMF)
#         #Test!
#         #-Does NMF/UMAP give better results with or without ROI detection?
#         #-Does local calibration give better results per scan or per binned scans or total (bin all) 
    
#     #""+1
    
#     #%% Figure out how to do map reduce in python
    
#     #%%detect ROI
    
    
#     ### Local calibration ###
    
#     #sum per pixels#
#     #try to deconvolute linear signal
    
# #%%
#     ""+1
#     ### Per scan analysis #### 

#     ds= ds[np.lexsort((ds[:,3],ds[:,2],ds[:,1],ds[:,5])) ]  #group binned pixels
    
    
#     ### construct peak intervals 
#     mp=c2m(p,sf,k0)
#     calr=m2c(pd.DataFrame([[i,i*(1-ppm_cal/1e6),i*(1+ppm_cal/1e6)] for i in mp],
#                           columns=["cal_channel","cal_lower","cal_upper"]),sf,k0)
#     calr["interval"]=calr.apply(lambda x: np.arange(x["cal_lower"],x["cal_upper"]+1),axis=1)
#     calr=calr.explode("interval")
#     calr=calr[["cal_channel","interval"]].astype(int).set_index("interval")
#     calr=calr.groupby(calr.index,sort=False).nth(0)
#     tcalr=calr[calr.cal_channel.isin(tp)]
    
#     pds=ds[np.in1d(ds[:,3].astype(int),tcalr.index.values.astype(int))]  # select only channels within top peak intervals
#     dss=np.array_split(pds,np.argwhere(np.diff(pds[:,5])>0)[:,0]+1)      # split on scans
#     if max_scans: dss=dss[:int(max_scans/bin_scans)]      
    
#     if Calibrate_local:
#         if Calibrate_global: fCalibrants=qcal #re-use same calibrants list
#         else: fCalibrants=np.sort(np.array([pySPM.utils.get_mass(i) for i in Calibrate_local if i.endswith(mode_sign)])) 

#     ### Per scan analysis
#     totals=np.zeros(number_channels)
#     ppm_map,rois,cdfs,kdfs=[],[],[],[]
    
#     b=np.nan
    
#     for d_ix,d in enumerate(dss):
        
#         print(d_ix)
#         sl=np.vstack([np.argwhere(np.diff(d[:,1])>0),np.argwhere(np.diff(d[:,2])>0)])[:,0]+1
#         sl.sort()
#         pixs=np.array_split(d[:,3],sl)
        
       
#         #### Calibrate local ####
    
#         if Calibrate_local:
#             mppms,cmpixs=[],[]
#             for ip,pix in enumerate(pixs):

#                 mpix=c2m(pix,sf,k0)
#                 closest=np.array([take_closest(mpix,i) for i in fCalibrants])
#                 ppms=(closest-fCalibrants)/ fCalibrants*1e6
#                 ppms=ppms[abs(ppms)<ppm_cal]
                
               
                
#                 if len(ppms): 
#                     best=ppms[np.argmin(abs(ppms))]
#                     if abs(best)<ppm_cal: 
#                         mppms.append(best)
#                         continue
                        
#                 mppms.append(b)
                
#             #store 2d ppm map
#             lcal=pd.DataFrame(d[np.hstack([0,sl])][:,1:3],columns=["x","y"])
#             lcal["ppm"]=mppms
#             pv=lcal.pivot(columns="x",index="y")
#             pv=pv.fillna(0)
#             pv.loc[:,:]=savgol2d(pv.values,window_size=5,order=3)
#             pv.columns=pv.columns.droplevel()
#             ppm_map.append(pv)
            
#             #map back
#             mpv=pv.melt()
#             mpv.columns=["x","ppm"]
#             mpv["y"]=pv.columns.astype(int).tolist()*len(pv.index)
#             mpv["x"]=mpv["x"].astype(int)
#             mpv["bin_scan"]=d_ix
            
#             cdf=pd.DataFrame(d[:,1:3].astype(int),columns=["x","y"]).merge(mpv,on=["x","y"],how="left")
#             cdfs.append(cdf.drop_duplicates())
#             d[:,3]=m2c(c2m(d[:,3],sf,k0)*(1-cdf.ppm/1e6).values,sf,k0).astype(int)
#             d=d[(d[:,3]<number_channels) & (d[:,3]>0)]   #error from ppm calibration 

#         #### detect ROI ####

#         #map peaks
#         pks=d[np.in1d(d[:,3].astype(int),tcalr.index.astype(int)),1:4].astype(int)
#         pks[:,2]=tcalr.loc[pks[:,2]].values[:,0]
#         kdf=pd.DataFrame(pks,columns=["x","y","peak"]).groupby(["x","y","peak"]).size()
#         kdf=kdf.reset_index().pivot(columns="peak",index=["x","y"]).fillna(0)
#         kdf.columns = kdf.columns.droplevel()  
#         kdfs.append(kdf)
#         totals[kdf.columns]=totals[kdf.columns]+1 #add totals


#     if Calibrate_local:
    
#         cdfs=pd.concat(cdfs)
#         cd=pd.DataFrame(ds[:,[1,2,5]].astype(int),columns=["x","y","bin_scan"]).merge(cdfs,on=["x","y","bin_scan"],how="left").fillna(0)
#         ds[:,3]=m2c(c2m(ds[:,3],sf,k0)*(1-cd.ppm/1e6).values,sf,k0).astype(int)
#         #ds[:,3]=m2c(c2m(ds[:,3],sf,k0)*(1+cd.ppm/1e6).values,sf,k0).astype(int)
#         ds=ds[(ds[:,3]<number_channels) & (ds[:,3]>0)]   #error from ppm calibration 
        
#         #plot summed 2d calibration map
#         cal2d = reduce(lambda x, y: x.add(y, fill_value=0), ppm_map)/scans*bin_scans
#         fig,ax=plt.subplots()
#         sns.heatmap(cal2d)
#         plt.title(Path(itmfile).stem)
#         fig.savefig(itmfile.replace(Path(itmfile).suffix,"_tb"+str(bin_tof)+"_pb"+str(bin_pixels)+"_2dcal_map.png"),bbox_inches="tight",dpi=300)
    
    
#         #%% Plot effects of local calibration
        
#         if bool(Calibrate_local) & bool(Calibrate_local):
        
#             u,uc=np.unique(ds[:,3],return_counts=True)#,return_inverse=True)
#             Spectrum=nan_arr
#             Spectrum[u.astype(int)]=uc
#             Spectrum=pd.Series(Spectrum).interpolate().fillna(0).rolling(window=smoothing_window).mean()  #rolling mean will cause slight shifts in peak location.
#             p,pi=find_peaks(Spectrum,prominence=prominence,distance=distance) 
#             tp=p[np.sort(np.argsort(pi["prominences"])[-top_peaks:])]
#             mp=c2m(tp,sf,k0)
    
#             #get calibrants
#             ppms=abs((qcal-mp.reshape(-1,1))/mp.reshape(-1,1))*1e6
#             q=(ppms<=ppm_cal).any(axis=1)
#             qmp=mp[q] 
#             qmp=np.array([take_closest(qmp,i)  for i in qcal])
            
        
#             post_local_ppms=(qmp-qcal)/qcal*1e6
#             q=abs(post_local_ppms)<ppm_cal
            
#             post_local_ppms=post_local_ppms[q]
#             post_ppms=post_ppms[q]
#             qcal=qcal[q]
            
#             posttl=str(round(sum(abs(post_local_ppms))/len(post_local_ppms),1))
#             postt=str(round(sum(abs(post_ppms))/len(post_ppms),1))
            
#             #plotting
#             fig,ax=plt.subplots()
#             plt.scatter(qcal,post_ppms,c=(0.5, 0, 0, 0.3))
#             plt.scatter(qcal,post_local_ppms,c=(0, 0.5, 0, 0.3))
#             plt.legend(["pre local calibration","post local calibration"])
#             plt.xlabel("m/z")
#             plt.ylabel("ppm mass error")
#             plt.title("local_calibration")
#             fig.savefig(itmfile.replace(Path(itmfile).suffix,"_tb"+str(bin_tof)+"_pb"+str(bin_pixels)+"_loc_cal_scat.png"),bbox_inches="tight",dpi=300)

#             fig,ax=plt.subplots()
#             y1, _, _ =plt.hist(post_ppms,color=(0.5, 0, 0, 0.3))
#             y2, _, _ =plt.hist(post_local_ppms,color=(0, 0.5, 0, 0.3))
#             plt.vlines(np.mean(post_ppms),0,np.hstack([y1,y2]).max(),color=(0.5, 0, 0, 1),linestyle='dashed')
#             plt.vlines(np.mean(post_local_ppms),0,np.hstack([y1,y2]).max(),color=(0, 0.5, 0, 1),linestyle='dashed')
#             plt.xlabel("ppm mass error")
#             plt.ylabel("frequency")
#             plt.legend(["pre: mean "+str(round(np.mean(post_ppms),1))+ ", abs "+postt,
#                         "post: mean "+str(round(np.mean(post_local_ppms),1))+ ", abs "+posttl],
#                         loc=[1.01,0])
#             plt.title("local_calibration")
#             fig.savefig(itmfile.replace(Path(itmfile).suffix,"_tb"+str(bin_tof)+"_pb"+str(bin_pixels)+"_loc_cal_hist.png"),bbox_inches="tight",dpi=300)

#             ""+1
#         # #plot indifidual maps
#         # for m in ppm_map:
#         #     fig,ax=plt.subplots()
#         #     sns.heatmap(m)
        
#     #%% Scan total clustering on chemical signatures for ROI detection
#     minsize=5
#     max_channels=500 #otherwise kmeans can get memory overloaded
#     allowed_channels=np.argwhere(totals>=minsize)[:,0]
    
#     fkdfs=[]
#     for i in kdfs:
#         fkdf=i[i.columns[i.columns.isin(allowed_channels)]]
#         if len(fkdf):
#             fkdfs.append(fkdf)
    
#     t=pd.concat(fkdfs).fillna(0) #concat is slow probably faster to flatten kdf and sum on flattened index
#     ts=t.groupby(t.index).sum()
#     ts[ts<minsize]=0
#     ts=ts[ts.columns[(ts>0).any(axis=0)]]
#     ts=ts[ts.columns.sort_values()]
    
#     nts=ts.divide(ts.sum(axis=1).values,axis=0)
#     nts[["x","y"]]=nts.index.tolist()
    
#     if len(nts.columns)>max_channels:
#         nts=nts[nts.sum().sort_values()[::-1][:max_channels].index.tolist()]

#     try:
#         kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(ts.values)
#         nts["roi"]=kmeans.labels_
        
#         hm=nts[["x","y","roi"]].pivot(columns="x",index="y")#.fillna(0) #fill with fillna or with no_clusters+1
#         hm.columns = hm.columns.droplevel()
        
#         fig,ax=plt.subplots()
#         sns.heatmap(hm)
#         plt.title(Path(itmfile).stem)
#         hm.to_csv(itmfile.replace(Path(itmfile).suffix,"_tb"+str(bin_tof)+"_pb"+str(bin_pixels)+"_ROI_map.tsv"),sep="\t")
#         fig.savefig(itmfile.replace(Path(itmfile).suffix,"_tb"+str(bin_tof)+"_pb"+str(bin_pixels)+"_ROI_map.png"),bbox_inches="tight",dpi=300)
    
#     except:
#         nts["roi"]=0
#         print("ROI detection failed!")
#     #%% Get depth profiles


#     print("Depth profile")
#     pds=ds[np.in1d(ds[:,3].astype(int),calr.index.values.astype(int))]  # select only channels within peak intervals
#     ddf=pd.DataFrame(pds,columns=["scan","x","y","tof","w","scan_bin"])
#     ddf[["x","y"]]=ddf[["x","y"]].astype(int)
#     ddf=ddf.set_index(["x","y"])
    
#     dps=[]

#     #%%
#     #sulphur plot
#     fig,ax=plt.subplots()
#     mp=c2m(np.arange(number_channels),sf,k0)
#     ms=( mp>31.95) & ( mp<32.05)
    
#     lines=[]
#     for cl in range(n_clusters):
        
#         xy=nts.loc[nts["roi"]==cl,["x","y"]].values.astype(int)
#         if not len(xy): continue
#         roi=ddf.merge(pd.DataFrame(xy,columns=["x","y"]),on=["x","y"],how="inner")
        
    
#         s=roi.groupby("tof")["w"].sum()
#         Spectrum=np.zeros(number_channels)
#         Spectrum[s.index.astype(int)]=s.values
        
#         #Save spectrum
#         s=pd.Series(Spectrum)
#         s=s[s>0]
#         s.index=np.round(c2m(s.index,sf,k0),3)
#         s=s.groupby(s.index).sum()
#         s.to_csv(itmfile.replace(Path(itmfile).suffix,"_tb"+str(bin_tof)+"_pb"+str(bin_pixels)+"_ROI"+str(cl)+"_Spectrum.tsv"),sep="\t")
        
        
        
#         Spectrum=pd.Series(Spectrum).interpolate().fillna(0).rolling(window=smoothing_window).mean().bfill()
        

#         #construct intervals
#         p,pi=find_peaks(Spectrum,prominence=prominence,distance=distance)
        
#         pidf=pd.DataFrame(pi)
#         pidf["width"]=pidf["right_bases"]-pidf["left_bases"]
#         pidf["fwhm"]=peak_widths(Spectrum, p, rel_height=0.5)[0]
#         pidf["intensity"]=Spectrum[p].values
#         pidf["mass"]=c2m(p,sf,k0)
        
#         pidf["cal_channel"]=p
#         pidf["cal_lower"]=np.floor(pidf["cal_channel"]-pidf["fwhm"]/2)
#         pidf["cal_upper"]=np.ceil(pidf["cal_channel"]+pidf["fwhm"]/2)
#         calr=pidf[["cal_channel","cal_lower","cal_upper"]]
        

#         calr["interval"]=calr.apply(lambda x: np.arange(x["cal_lower"],x["cal_upper"]),axis=1)
#         #calr["interval"]=calr.apply(lambda x: np.arange(x["cal_lower"],x["cal_upper"])+1,axis=1)
#         calr=calr.explode("interval")
#         calr=calr[["cal_channel","interval"]].astype(int).set_index("interval")
#         calr=calr.groupby(calr.index,sort=False).nth(0) 

        
        
#         roi=roi[np.in1d(roi.tof.astype(int).values,calr.index)]
        
    
#         #sulphur plot
#         line=ax.plot(c2m(np.arange(number_channels),sf,k0)[ms],Spectrum[ms],label="ROI "+str(cl))
#         lines.append(line)
        
        
#         roi=roi[["scan","x","y","tof","w"]].sort_values(by="scan")#.values 
#         dp=roi.groupby(["scan","tof"])["w"].sum().reset_index() ##.pivot(columns="scan",index="tof")
#         dp["tof"]=calr.loc[dp["tof"].astype(int)].values
        
        
#         g=dp.groupby(["scan","tof"])["w"]
        
#         Area=g.sum().reset_index().pivot(columns="scan",index="tof").fillna(0).astype(int)
#         Area.columns=Area.columns.droplevel()
#         Area.index=np.round(c2m(Area.index,sf,k0),3)
#         Area.to_csv(itmfile.replace(Path(itmfile).suffix,"_tb"+str(bin_tof)+"_pb"+str(bin_pixels)+"_ROI"+str(cl)+"_depth_profile_Area.tsv"),sep="\t")

#         Int=g.max().reset_index().pivot(columns="scan",index="tof").fillna(0).astype(int)
#         Int.columns=Int.columns.droplevel()
#         Int.index=np.round(c2m(Int.index,sf,k0),3)
#         Int.to_csv(itmfile.replace(Path(itmfile).suffix,"_tb"+str(bin_tof)+"_pb"+str(bin_pixels)+"_ROI"+str(cl)+"_depth_profile_Int.tsv"),sep="\t")
#         #find faster way than to csv, this is too slow

#     plt.legend()
#     plt.title(Path(itmfile).stem)
#     fig.savefig(itmfile.replace(Path(itmfile).suffix,"_tb"+str(bin_tof)+"_pb"+str(bin_pixels)+"_sulphurplot.png"),bbox_inches="tight")


