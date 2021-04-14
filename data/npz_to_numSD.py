'''
Calvin Pash
4/14/2021
.npz to num SDs

Takes stress and orientation data from .npz files in data/interim
Creates files in data/processed containing number of voxels with stress greater than Mean + n*SD
'''
import numpy as np
import os
from sys import argv

def main(args):
    from os import listdir
    sd = 2
    for arg in args:#Takes in command line arguments; less sugar but more rigorous
        try: arg = int(arg)
        except: pass
        if type(arg) == int:
            sd = arg
        else:
            print(f"Argument '%s' ignored" % str(arg))

    all_dat = np.array([np.array(list(np.load(f"data/interim/{i}").values())) for i in os.listdir("data/interim")])

    stress_mean = np.mean(all_dat[:,3,:,:,:])
    stress_sd = np.std(all_dat[:,3,:,:,:])
    #stress_by_layers = all_dat[:,3,:,:,:].reshape()

    shape_by_layer = (all_dat.shape[2]*all_dat.shape[0],all_dat.shape[3], all_dat.shape[4])
    stress_by_layer = all_dat[:,3,:,:,:].reshape(shape_by_layer)

    stress_num_hotspot = np.array([np.sum(np.logical_or(layer < stress_mean - sd*stress_sd, stress_mean + sd*stress_sd < layer)) for layer in stress_by_layer])

    print(f"\tMean\tSD")
    print(f"\t%.2f\t%.2f" % (np.mean(stress_num_hotspot), np.std(stress_num_hotspot)))
    #ori_by_layers = all_dat[:,:3,:,:,:].reshape(all_dat.shape[2]*all_dat.shape[0],all_dat.shape[3], all_dat.shape[4],3)




if __name__ == '__main__':
    main(argv[1:])
