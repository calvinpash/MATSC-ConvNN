'''
Calvin Pash
4/14/2021
.npz to num SDs

Takes stress and orientation data from .npz files in data/interim
Creates files in data/processed containing number of voxels with stress greater than Mean + n*SD
'''
import numpy as np
import os
from sys import argv, exit
import matplotlib.pyplot as plt

def main(args):
    from os import listdir
    sd = 2
    for arg in args:#Takes in command line arguments; less sugar but more rigorous
        try: arg = float(arg)
        except: pass
        if type(arg) == float:
            sd = arg
        else:
            print(f"Argument '%s' ignored" % str(arg))

    dat_dir = "data/interim"
    all_dat = np.array([np.array(list(np.load(f"{dat_dir}/{i}").values())) for i in os.listdir(dat_dir) if i.endswith("npz")])
    
    all_dat = all_dat[:,:, 1:-1, 1:-1, :]
    
    #shape: (file, channel, x, y, z)
    
    all_dat = np.moveaxis(all_dat, 4, 2)

    #shape: (file, channel, z, x y)
    
    stress_mean = np.mean(all_dat[:,3,:,:,:])
    stress_sd = np.std(all_dat[:,3,:,:,:])
    
    all_dat = np.moveaxis(all_dat,1,4)
    
    #plt.imshow(all_dat[0,24,:,:,0]).write_png("all_dat_ori1.png")
    #plt.imshow(all_dat[0,24,:,:,1]).write_png("all_dat_ori2.png")
    #plt.imshow(all_dat[0,24,:,:,2]).write_png("all_dat_ori3.png")
    #plt.imshow(all_dat[0,24,:,:,:3]+.5).write_png("all_dat_ori.png")
    #plt.imshow(all_dat[0,24,:,:,3]).write_png("all_dat_stress.png")
    
    shape_by_layer = (all_dat.shape[0]*all_dat.shape[1], all_dat.shape[2], all_dat.shape[3], all_dat.shape[4])
    dat_by_layer = all_dat.reshape(shape_by_layer)

    #Fit stress values between 0 and 1
    stress_norm = np.array([(layer-layer.min())/(layer.max()-layer.min()) for layer in dat_by_layer[:,:,:,3]])

    #plt.imshow(stress_norm[24]).write_png("norm_stress.png")
    
    np.savez("data/processed/processed_field.npz", dat_by_layer[:,:,:,:3], stress_norm)  


if __name__ == '__main__':
    main(argv[1:])
