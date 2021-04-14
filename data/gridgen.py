'''
Calvin Pash
2/2/21
Grid Generator
Takes in start, stop, spacing for x, y, and z coordinates and outputs list of space delimited grid points
'''


#%%
from sys import argv
import numpy as np

spacing=0.01
#Defines how granular
#This is in microns

x_start=0.
x_end=1.

y_start=0.
y_end=1.

#select the middle millimeter
z_start=1.
z_end=2.

x=np.arange(x_start, x_end, spacing) 
y=np.arange(y_start, y_end, spacing) 
z=np.arange(z_start, z_end, spacing) 

#gives a 3-D matrix of x positions, y positions, and z positions
#works the same was as np.reshape
X, Y, Z = np.meshgrid(x,y,z)    

#convenient way to reconstruct the grid
np.savez('./data/grid.npz',X=X,Y=Y,Z=Z)
np.savetxt('./data/grid.txt',np.vstack((X.ravel(),Y.ravel(),Z.ravel())).T)



