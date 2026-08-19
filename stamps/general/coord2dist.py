# -*- coding:utf-8 -*-
import numpy as np
import multiprocessing as mp
CPU_COUNT = mp.cpu_count()

from six.moves import range
from scipy import sparse
from scipy.spatial import cKDTree as KDTree
from scipy.spatial.distance import cdist


def coord2dist_mp( c1, c2, workers=max(1, CPU_COUNT - 1) ):
    '''
    workers    int    number of workers
    '''

    if c1.shape[0] * c2.shape[0] <= 10 ** 7 or workers == 1:
        return coord2dist(c1, c2)
    else:
        print('used worker: {w}'.format(w = workers))
        d_c1 = np.int(np.ceil(c1.shape[0] / float(workers)))
        d_c2 = np.int(np.ceil(c2.shape[0] / float(workers)))
        c12 = [(c1[d_c1 * i:d_c1 * (i + 1)],
                c2[d_c2 * i:d_c2 * (i + 1)]) for i in range(workers)]
        res_ij = [(c12[i][0], c12[j][1]) for i in range(workers)
                  for j in range(workers)]
        pool = mp.Pool(processes=workers)
        d = pool.map(_warp_coord2dist, res_ij)

        d = [np.vstack(d[i::workers]) for i in range(workers)]
        d = np.hstack(d)
        return d
    return d

def _warp_coord2dist(args):
    return coord2dist(*args)

def coord2dist( c1, c2, norm=2 ):
    '''
    Calculate the distance between coordinates c1 and c2
    
    Syntax: result = coord2dist(c1, c2)

    Input 
        c1        [r1 x d]     np.array of coordinates
        c2        [r2 x d]     np.array of coordinates
    Output
        d         [r1 x r2]    np.array of distances
    '''

    try:
        d = cdist(c1, c2, 'minkowski', p=norm)  
    except:
        ones_c1 = np.ones((c1.shape[0], 1))
        ones_c2 = np.ones((c2.shape[0], 1))
        a = np.kron(c1, ones_c2)
        b = np.kron(ones_c1, c2)
        d = ((a - b) ** 2)
        d = d.sum(axis=1)
        d = np.sqrt(d.astype(float))
        d = d.reshape((c1.shape[0], c2.shape[0]))
    return d
