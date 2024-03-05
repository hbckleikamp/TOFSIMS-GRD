# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 13:36:18 2023

@author: hkleikamp
"""
### Modules ###
from pathlib import Path
import os
from inspect import getsourcefile

# change directory to script directory (should work on windows and mac)
os.chdir(str(Path(os.path.abspath(getsourcefile(lambda:0))).parents[0]))
sippy_dir=os.getcwd()
basedir=os.getcwd()


import pySPM
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import struct
from collections import Counter
import scipy
#%% Read erics fragment list

fragment_files=["C:/Users/hkleikamp/Downloads/41467_2021_24312_MOESM6_ESM.xls",
"C:/Users/hkleikamp/Downloads/41467_2021_24312_MOESM5_ESM.xls"]

ffs=[]
for ff in fragment_files:
    fdf=pd.read_excel(ff,header=None)#,engine="openpyxl")
    fdf.columns=fdf.iloc[1,:]
    fdf=fdf.iloc[4:].reset_index()
    fdf=fdf[fdf["Expected Mass"].notnull()]
    
    #fdf=fdf[['ToF-SIMS assignment', 'Center Mass']]
    ppm_shifts=abs((fdf["Center Mass"]-fdf["Expected Mass"])/fdf["Expected Mass"]*1e6)
    print("meddian ppm shift : "+str(ppm_shifts.median()))
    
    
    fdf=fdf[['ToF-SIMS assignment', 'Center Mass']]
    fdf["polarity"]=fdf['ToF-SIMS assignment'].str[-1]
    ffs.append(fdf)
ffs=pd.concat(ffs).set_index("ToF-SIMS assignment")

ppm_tol=50
proton_mass=1.00727647
electron_mass=0.000548579909

#%%
ITM_files=["E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/N_Fiber sheets_3D_0,1nA_Bi3_4.itm",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/N_Fiber sheets_3D_1nA_1.itm",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/N_Fiber sheets_3D_1nA_2.itm",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/N_Fiber sheets_3D_1nA_Bi3_3.itm",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/N_Fiber sheets_3D_1pA.itm",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/N_Fiber sheets_surface.itm",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/N_Fiber sheets_surface2.itm",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/N_Poly Ni ett_Ar1500_20keV.itm",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/N_Poly Ni ett_Bi3.itm",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/N_Poly Ni ett_Bi5_1.itm",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/N_Poly Ni ett_Bi5_2.itm",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/P_Fiber sheets_3D_0,1nA_Bi3_1.itm",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/P_Fiber sheets_3D_0,1nA_Bi3_2.itm",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/P_Fiber sheets_3D_0,1nA_Bi3_3.itm",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/P_Fiber sheets_3D_0,1nA_Bi3_4.itm",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/P_Fiber sheets_3D_0,1nA_Bi3_5.itm",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/P_Fiber sheets_surface.itm",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/P_Poly Ni ett_Ar1500_20keV.itm",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/P_Poly Ni ett_Bi3.itm",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/P_Poly Ni ett_Bi5_1.itm",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/P_Poly Ni ett_BI5_2.itm"]

ITA_files=["E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/N_Fiber sheets_3D_0,1nA_Bi3_4_0.ita",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/N_Fiber sheets_3D_1nA_1_0.ita",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/N_Fiber sheets_3D_1nA_2_0.ita",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/N_Fiber sheets_3D_1nA_Bi3_3_0.ita",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/N_Fiber sheets_3D_1pA_0.ita",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/N_Fiber sheets_surface_0.ita",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/N_Fiber sheets_surface2_0.ita",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/N_Poly Ni ett_Ar1500_20keV_0.ita",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/N_Poly Ni ett_Bi3_0.ita",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/N_Poly Ni ett_Bi5_1_0.ita",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/N_Poly Ni ett_Bi5_2_0.ita",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/P_Fiber sheets_3D_0,1nA_Bi3_1_0.ita",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/P_Fiber sheets_3D_0,1nA_Bi3_2_0.ita",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/P_Fiber sheets_3D_0,1nA_Bi3_3_0.ita",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/P_Fiber sheets_3D_0,1nA_Bi3_4_0.ita",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/P_Fiber sheets_3D_0,1nA_Bi3_5_0.ita",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/P_Fiber sheets_surface_0.ita",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/P_Poly Ni ett_Ar1500_20keV_0.ita",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/P_Poly Ni ett_Bi3_0.ita",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/P_Poly Ni ett_Bi5_1_0.ita",
"E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/P_Poly Ni ett_Bi5_2_0.ita"]






ITM_files=["E:/Antwerp/ToF-SIMS Hugo/I180124a_C2sp1_pos.itm",
            "E:/Antwerp/ToF-SIMS Hugo/I180125a_C2sp2_neg.itm"]

ITA_files=["E:/Antwerp/ToF-SIMS Hugo/I180124a_C2sp1_pos_Roi2 3.ita",
            "E:/Antwerp/ToF-SIMS Hugo/I180125a_C2sp2_neg_Roi1 5.ita"]

ITM_files.sort()
ITA_files.sort()

#Open iontof files
#plot surface


for file in ITM_files:
    break



#%% Test itmx


        
#%%
filename="E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/N_Fiber sheets_3D_0,1nA_Bi3_4.itm"
filename="E:/Antwerp/ToF-SIMS Louvain/N_Fiber_sheets_3D_0,1nA_Bi3_4_0_(0)_-_total/N_Fiber sheets_3D_0,1nA_Bi3_4.itmx"
f = open(filename, 'rb')
s=str(f.read()).split("\\x19\\x00\\x00\\x00") #shouldn't always split but most of the time
#"i"+1


# l=[len(i) for i in s]
# fig,ax=plt.subplots()
# line, = ax.plot(l)

# ax.set_yscale('log')

#%%

class ITMX():
    def __init__(self):
        self.filename=filename
        
filename="E:/Antwerp/ToF-SIMS Louvain/N_Fiber_sheets_3D_0,1nA_Bi3_4_0_(0)_-_total/N_Fiber sheets_3D_0,1nA_Bi3_4.itmx"
#filename="E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/N_Fiber sheets_3D_0,1nA_Bi3_4.itm"

I=ITMX()


#I = pySPM.ITM("E:/Antwerp/ToF-SIMS Louvain/N_Fiber_sheets_3D_0,1nA_Bi3_4_0_(0)_-_total/N_Fiber sheets_3D_0,1nA_Bi3_4.itmx")

f = open(filename, 'rb')
I.f=f
print(f.read(8))

import binascii
l=[]

c=0
while True:
    
    try:
        c+=1
        I.type=binascii.hexlify(f.read(1)).decode("utf-8")
        I.b4=str(f.read(4))    
    
        I.head = dict(zip(['name_length', 'ID', 'N', 'length1', 'length2'], \
                             struct.unpack('<5I', I.f.read(20))))
        I.name = I.f.read(I.head['name_length']).decode('ascii')
        I.value = I.f.read(I.head['length1'])
        l.append([I.type,I.b4,I.name]+[I.head.get(x) for x in ['name_length', 'ID', 'N', 'length1', 'length2']] )
        
        # if I.name=="SIMSDataSet":
        #     break
        
        
        print(f.tell())
    except:
        break
bdf=pd.DataFrame(l,columns=["type","b4","name","name_length","ID","N","l1","l2"])
# #b=pySPM.Block.Block(r)
# #r.read(5)

# # self.Type = self.f.read(8)

# for x in range(10):
#     offset = I.offset
#     I.f.seek(offset)
#     I.head = dict(zip(['name_length', 'ID', 'N', 'length1', 'length2'], struct.unpack('<5I', I.f.read(20))))
#     I.name = I.f.read(I.head['name_length'])

#     l.append([I.head.get(x) for x in ['name_length', 'ID', 'N', 'length1', 'length2']] )
#     I.offset = I.f.tell()
# Block.Block(self.f)

#%%
for file_ix,file in enumerate(ITA_files):

    # if file=="E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/P_Fiber sheets_3D_0,1nA_Bi3_1_0.ita":
    #     break
    
    fs=Path(file).stem
    if fs=="N_Poly Ni ett_Bi3_0_depth_profile":
        break
    
    break
    
   

#%%


    I = pySPM.ITM(ITM_files[file_ix])
    A = pySPM.ITA(file)
    
    


    # @alias("getSpectrum")
    # def get_spectrum(self, sf=None, k0=None, scale=None, time=False, error=False, **kargs):
    #     """
    #     Retieve a mass,spectrum array
    #     This only works for .ita and .its files.
    #     For this reason it is implemented in the itm class.
    #     """
    #     RAW = zlib.decompress(self.root.goto(
    #         'filterdata/TofCorrection/Spectrum/Reduced Data/IITFSpecArray/' + ['CorrectedData', 'Data'][
    #             kargs.get('uncorrected', False)]).value)
    #     if scale is None:
    #         scale = self.scale
    #     D = scale * np.array(struct.unpack("<{0}f".format(len(RAW) // 4), RAW))
    #     ch = 2 * np.arange(len(D))  # We multiply by two because the channels are binned.
    #     if time:
    #         return ch, D
    #     m = self.channel2mass(ch, sf=sf, k0=k0)
    #     if error:
    #         # TODO: parameters Dk0 and Dsf are undefined
    #         Dm = 2 * np.sqrt(m) * np.sqrt(Dk0 ** 2 + m * Dsf ** 2) / sf
    #         return m, D, Dm
    #     return m, D
    
#%%
# import zlib

# A.show_spectrum() #works
# A.get_spectrum() #works

# RAW = zlib.decompress(A.root.goto('filterdata/TofCorrection/Spectrum/Reduced Data/IITFSpecArray/Data').value)
# D = np.array(struct.unpack("<{0}f".format(len(RAW) // 4), RAW))
    #%%
    ## Plot snapshot
    meta_S=A.root.goto('Meta/Video Snapshot').dict_list()
    meta_I=A.root.goto('Meta/SI Image').dict_list()
    
    sx,  sy  =meta_S['res_x']['ulong']              ,meta_S['res_y']['ulong']               #pixels
    x_um,y_um=meta_S["fieldofview_x"]["float"]*10**6,meta_S["fieldofview_y"]["float"]*10**6 #real FOV
    zoom_factor=meta_I["zoomfactor"]["float"]
    scale_x, scale_y = x_um/sx, y_um/sy  #pixels to len
    xlen=A.size["real"]["x"]*1e6/zoom_factor/scale_x #in pixels
    ylen=A.size["real"]["y"]*1e6/zoom_factor/scale_y #in pixels
    

    try:
        img = np.array(A.root.goto('Meta/Video Snapshot/imagedata').get_data()).reshape((sy, sx, -1)) #error in module
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1) 
        ax.imshow(img)
        plt.title(fs)
    
        
        
    
        plt.xticks(ax.get_xticks(),(ax.get_xticks()*scale_x).astype(int))
        plt.yticks(ax.get_yticks(),(ax.get_yticks()*scale_y).astype(int))
        plt.xlim(0,sx)
        plt.ylim(0,sy)
    
        #add scan patch
        
        xdiff=(meta_S['stageposition_x']["float"]-meta_I['stageposition_x']["float"])*1e6/scale_x
        ydiff=(meta_S['stageposition_y']["float"]-meta_I['stageposition_y']["float"])*1e6/scale_y

        

        
    # #works for "E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/P_Fiber sheets_3D_0,1nA_Bi3_5_0.ita"
    #     xstart=(sx-xlen-xdiff)/2
    #     ystart=(sy-ylen-ydiff)/2
    
    # # #works for "E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/P_Fiber sheets_3D_0,1nA_Bi3_4_0.ita"
    #     xstart=(sx-xlen)/2
    #     ystart=(sy-ylen)/2
        
        # #works for "E:/Antwerp/ToF-SIMS Louvain/20231107_cablebacteria/P_Fiber sheets_3D_0,1nA_Bi3_3_0.ita"
        xstart=(sx-xlen)/2-xdiff
        ystart=(sy-ylen)/2-ydiff
            
    
        plt.scatter(sx/2,sy/2,s=5,c="red",marker="x")
        
        rect = patches.Rectangle((xstart,ystart), xlen, ylen, linewidth=1, edgecolor='black', facecolor='none',linestyle="--")
        ax.add_patch(rect)
        
        plt.xlabel(u"\u03bcm")
        plt.ylabel(u"\u03bcm")
        plt.legend(["center","SI-zone"],loc=[1.02,0.45])
      
        fig.savefig(fs+"_snapshot.png",dpi=1000)
        
    except:
        pass
    
    ##### plot SI
    X, Y = A.size['pixels']['x'], A.size['pixels']['y']
    arr=np.array(A.root.goto('Meta/SI Image/intensdata').get_data("f"), dtype=np.float32).reshape((Y, X))


  

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1) 
    i=ax.imshow(arr,cmap="Greys_r")
    plt.colorbar(i)
    plt.title(fs)

    plt.xticks(ax.get_xticks(),(ax.get_xticks()/ax.get_xticks().max()*xlen*scale_x).astype(int))
    plt.yticks(ax.get_yticks(),(ax.get_yticks()/ax.get_yticks().max()*ylen*scale_y).astype(int)[::-1])
    plt.xlim(0,X)
    plt.ylim(Y,0)

    plt.xlabel(u"\u03bcm")
    plt.ylabel(u"\u03bcm")

    fig.savefig(fs+"_SI.png",dpi=1000)

    

    #% Access depth profile
    #scans=np.arange(I.get_summary()["Scans"])
    scans=np.arange(I.Nscan)
    
    #%% test
    
    #dead time correction
    DT=0
    dts = DT * (I.size['pixels']['x'] / 2 - np.arange(
    I.size['pixels']['x']))  # time correction for the given x coordinate (in channel number)
    
    number_channels = int(round(A.root.goto('propend/Measurement.CycleTime').get_key_value()['float'] \
                                / A.root.goto('propend/Registration.TimeResolution').get_key_value()['float']))
    Spectrum = np.zeros(number_channels, dtype=np.float32)
    
    
    scans=np.arange(I.Nscan)
    for s in scans:
        print(s)
        raw = I.get_raw_raw_data(s)
        rawv = struct.unpack('<{}I'.format(len(raw) // 4), raw)
        i = 0
        while i < len(rawv):
            b = rawv[i]
            if b & 0xc0000000:
                x = b & 0x0fffffff
                dt = dts[x]
                ip = int(dt)
                fp = dt % 1
                i += 3
            else:
                Spectrum[b - ip] += (1 - fp)
                Spectrum[b - ip - 1] += fp
                i += 1
                
        break

    #%% test 2

    RAW = b''
    scan=0
    I.rawlist = I.root.goto('rawdata').get_list()
    startFound = False
    for x in I.rawlist:
        if x['name'] == '   6':
            if not startFound:
                startFound = x['id'] == scan
            else:
                break
        elif startFound and x['name'] == '  14':
            I.root.f.seek(x['bidx'])
            child = Block.Block(I.root.f)
            RAW += zlib.decompress(child.value)
            "i"+1

    #%% test3
    rawlist = I.root.goto('rawdata').get_list()
    raw_df=pd.DataFrame.from_dict(rawlist)
    for i in rawlist:
        if i['name'] == '   6':
            print("k")
            break
    

    #%%
    
    
    
    t = I.get_meas_data('Measurement.TotalTime')
    S = I.get_meas_data("Measurement.ScanNumber")
    print("no scans: "+str(S))
    idx = [x[0] for x in t]
    time = [x[1] for x in t]
    ScanIdx = [x[0] for x in S]
    scantimes = np.round(np.interp(ScanIdx, idx, time),1)
    
    number_channels = int(round(A.root.goto('propend/Measurement.CycleTime').get_key_value()['float'] \
                                / A.root.goto('propend/Registration.TimeResolution').get_key_value()['float']))
    
    channel_arr=np.array(range(number_channels))
    
    
    
    ds=[]
    for s in scans:
        
        print(s)
        try:
            raw = I.get_raw_raw_data(s)
            rawv = np.array(struct.unpack('<{}I'.format(len(raw) // 4), raw)).flatten()
        
            coords=np.argwhere(rawv>number_channels).flatten()
            data_startcoords[::3][1:] #=coords[np.argwhere(np.diff(coords)>1).flatten()][1:]-2
            #only exception is that sometimes there are completely empty pixels!
            
            pixs=np.array_split(rawv,data_start)
            xs,ys,ms=[],[],[]
            
            for c,p in enumerate(pixs):
                xs.append(p[0] & 0x0fffffff)
                ys.append(p[1] & 0x0fffffff)
                
                #check when this fails
                
                #ms.append(p[p<number_channels]) #how can it become larger than number of channels?
                
                ms.append(p[3:]) #how can it become larger than number of channels?
                
                
        except:
            print("Error, skipping scan")
    
        
        all_points=np.hstack(ms)
        
    
        
        
        counts=Counter(all_points.tolist())
        counts_df = pd.DataFrame.from_dict(counts, orient='index')
        
    
        ds.append(counts_df)
    
    ds=pd.concat(ds,axis=1).sort_index().fillna(0).astype(int)
    ds.columns=scantimes
    
    
    ds.index=I.channel2mass(ds.index)
    

    #%%
    
    if I.polarity=="Positive": 
        frags=ffs[ffs.polarity=="+"]
        frags["Center Mass"]-=electron_mass 
    else:
        frags=ffs[ffs.polarity=="-"]
        frags["Center Mass"]+=electron_mass
        
    rs=[]
    for n,frag in frags.iterrows():

    
        m=frag["Center Mass"]
        
        r=ds[(ds.index<m*(1+ppm_tol/1e6)) & (ds.index>m*(1-ppm_tol/1e6))].sum()
        rs.append(r)
    rs=pd.concat(rs,axis=1).T
    rs.index=frags.index
    rs.to_csv(fs+"_depth_profile.tsv",sep="\t")
    
        
    #plot
    # fig,ax=plt.subplots()
    # plt.plot(r)
    # plt.title(fs+" "+n)
    # plt.ylabel("counts")
    # plt.xlabel("time (s)")

    
    
    #%% Summary
    
    summ=I.get_summary()
    s=pd.Series(summ.values())
    s.index=summ.keys()
    s.to_csv(fs+"_summary.tsv",sep="\t",header=None)

    #%% manual checks
    
    # tm=57.935+electron_mass #+proton_mass not +H+ but -e
    
    # isotopes=[59.9307858]
    
    # m,i=A.get_spectrum()

 
    # dss=ds.sum(axis=1)
    # m,i=dss.index,dss.values

    # ll,ul=57.8,60.2
    # mNi=m[(m>ll) & (m<ul)]
    # iNi=i[(m>ll) & (m<ul)]
    
    # #pick peaks
    
    # # smoothed=padded_counts_df.rolling(window=5).mean().fillna(0)
    
    
    # # peaks,props=scipy.signal.find_peaks(smoothed.values.flatten(), height=None, threshold=None, 
    # #                         distance=1, prominence=5, width=5, 
    # #                         wlen=None, rel_height=0.5, plateau_size=None)
    
    
    # #get area under peaks
    # #add label of area
    
    # fig,ax=plt.subplots()
    # ax.vlines([tm]+isotopes,0,iNi.max(),color="red")
    # plt.plot(mNi,iNi)
    # plt.title("Ni+")
    
    # #add label to area
    # #plot isotopes
    

    #%% Heatmap plot of all fragments
        # #pad zeroes 
        # pad=set(channel_arr)-set(counts_df.index)  #this line can be coded faster
        # pad_df=pd.DataFrame([0]*len(pad),index=list(pad))
        # counts_df=pd.concat([counts_df,pad_df]).sort_index().astype(int)
    
    
    # print("padding")
    # ix,inc=ds.index,0
    # pad=[]
    # for i in np.arange(number_channels):
    #     if inc<len(ds):
        
    #         if i==ix[inc]:
    #             inc+=1
    #             continue
    #     pad.append(i)
    
    # pad_df=pd.DataFrame(np.zeros((len(pad),len(ds.columns))),index=pad,columns=ds.columns)
    # ds=pd.concat([ds,pad_df]).sort_index().astype(int)
    # ds.index=I.channel2mass(channel_arr)
    
    

    

    # from matplotlib.colors import LogNorm
    
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1) 
    # ax.ticklabel_format(useOffset=False)
    # i=ax.imshow(ds.values,aspect="auto", cmap='coolwarm',interpolation = 'sinc')#,cmap="Reds",norm=LogNorm(vmin=1, vmax=ds.max().max()))
    # plt.colorbar(i)
    # plt.title(fs)
    # #plt.colorbar()

    # plt.yticks(ax.get_yticks(),np.round(I.channel2mass(ax.get_yticks()),2))
    # plt.xlabel("Total time (s)")
    # plt.ylabel("Mass MH+")
    # plt.ylim(0,ax.get_yticks().max())



    #%% Questions claude
    
    #Field of view, what is sputtered, what is measured
    #How to align SI with snapshot
    
    #nAmperes or picoAmperes (should be Pico)
    #is Pico-measuerement correct, seems not positioned well?
    
    #Channel: intensity or binary
    
    #%% To Do
    #calibration
    #isotope check of fragments
    #analysis reports
    #row standardize + clustergrams for sorting rows
    #2D plots, regions of interest?
