import os
import numpy as np
fname='test.txt'

if os.path.exists(fname):
    data = np.loadtxt(fname)
    append_len=50-len(data)
    zeros = np.zeros((append_len,5))
    data = np.concatenate([data,zeros])
    np.savetxt('test.txt', data,fmt='%f',delimiter='\t')
