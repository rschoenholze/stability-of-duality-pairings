import matplotlib.pyplot as plt
import numpy as np


#write minEV tensor from computation files to a txt file, then read that file into this one and so the plotting

# for d in range(2,4):
#     for l in range(0,d+1):
#         print("d{d}l{l}".format(d=d,l=l))
#         data = np.load('d{d}l{l}_minEV'.format(d=d,l=l) + '.npy')

data = np.load('d{d}l{l}_minEV'.format(d=3,l=3) + '.npy')

print(data)