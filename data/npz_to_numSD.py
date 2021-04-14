'''
Calvin Pash
4/14/2021
.npz to num SDs

Takes stress and orientation data from .npz files in data/interim
Creates files in data/processed containing number of voxels with stress greater than Mean + n*SD
'''
import numpy as np
from sys import argv
