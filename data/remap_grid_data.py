#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 19:23:19 2021

@author: dcp5303
"""

#%%

import numpy as np
import matplotlib.pyplot as plt
from sys import argv, exit

#%%

num = argv[1]


data=np.load('data/grid.npz')


sim_val=1


els_file=f"data/meshes/result_{num}.stpoint"
oris_file=f'/tri-s-1/s2/dcp5303/projects/2021_03_16_GNN_Paper/test_{num}.sim/results/elements/ori/ori.step0'
stress_file=f'/tri-s-1/s2/dcp5303/projects/2021_03_16_GNN_Paper/test_{num}.sim/results/elements/stress/stress.step1'

#stress/stress.step1

#%%

els=np.loadtxt(els_file)
ori_raw=np.loadtxt(oris_file)
stress_raw=np.loadtxt(stress_file)

print(f"Running on {num}")
print(f"\t{els.shape}")
print(f"\t{ori_raw.shape}")
print(f"\t{stress_raw.shape}")

#%%

stresses=np.zeros(len(els))
oris_0=np.zeros(len(els))
oris_1=np.zeros(len(els))
oris_2=np.zeros(len(els))

for ii in np.arange(len(els)):    
    
    if els[ii] < 0:
        oris_0[ii]=0.0
        oris_1[ii]=0.0
        oris_2[ii]=0.0
        stresses[ii] =-1.0   
        
    else:
        oris_0[ii]=ori_raw[els[ii].astype(int)-1,0]
        oris_1[ii]=ori_raw[els[ii].astype(int)-1,1]
        oris_2[ii]=ori_raw[els[ii].astype(int)-1,2]
        stresses[ii]=stress_raw[els[ii].astype(int)-1,2]

ori0_mat=np.reshape(oris_0,data['X'].shape) 
ori1_mat=np.reshape(oris_1,data['X'].shape)  
ori2_mat=np.reshape(oris_2,data['X'].shape)   
stress_mat=np.reshape(stresses,data['X'].shape)    #dimensions are currently x y z

np.savez(f"./data/interim/dat_{num}.npz", ori0_mat, ori1_mat, ori2_mat, stress_mat)



#%%

plt.imshow(ori0_mat[:,:,10])

