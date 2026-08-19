# -*- coding: utf-8 -*-
import numpy as np

from .coord2dist import coord2dist
from .isspacetime import isspacetime
from ..models.covmodel import get_model


def coord2K(c1, c2, models, params):
    '''Compute the covariance or variogram matrix between two sets of coordinates,
    based on the Euclidean distances between these sets of coordinates.
    
    Args:
        c1 (ndarray): 2-D ndarray of floats with shape (m1,nd). S/T coordinates of m1 data.
        c2 (ndarray): 2-D ndarray of floats with shape (m2,nd). S/T coordinates of m2 data.
        models (list): List of string that contains a sequence of covariance models string.
        params (list): List of sequence of float that contains a sequence of covariance parameters values.

    Returns:
        Ki_sum (ndarray):2-D ndarray of floats with shape (m1,m2). 
            covariances between c1 and c2 datasets 
        Ki (list):list of m1 by m2 covariances from each of nested covariance models
    
    Note:
        Formats of covariance model and parameters  

        isST

            isSTsep
                models: ['exponentialC','exponentialC','...'] 
                params: [(3,None), (21.9, 35.8)] 
            not isSTsep
                models: ["gaussianCST", "exponentialCST", "..." ]
                params: [(sill1, bs1, stratio),
                         (sill2, bs2, stratio)]
        not isST
        
            models: ["gaussian", "exponential", "..."" ]
            params: [(sill1, bs1),
                     (sill2, bs2)] 
  
    Examples:
        Randomly create two coordinate sets
        >>> c1 = np.array([[10,15],[6,4],[-14,-6],[8,-4]])
        >>> c2 = np.array([[5,-5],[12,8],[0,-8],[6,10]])

        specify the covariance model and parameter.
        >>> models = ['exponentialC']
        >>> params = [(10,7)]

        Calculate covariance between two sets of coordinates
        >>> Ki_sum,Ki = coord2K(c1, c2, models, params) 
    '''
    if c1.size == 0 or c2.size == 0:
        Ki = []
        for model in models:
            Ki.append(np.array([]).reshape((c1.shape[0], c2.shape[0])))
        Ki_sum = sum(Ki)
        return Ki_sum, Ki
    
    isST, isSTsep, model_res = isspacetime(models)
    if isST:
        if isSTsep:
            modelS, modelT = model_res
            dist_s = coord2dist(c1[:, 0:2], c2[:, 0:2])
            dist_t = coord2dist(c1[:, 2:3], c2[:, 2:3])
            Ki = []
            for model_s, model_t, param_i in zip(modelS, modelT, params):
                sill, param_s, param_t = param_i
                model_s = get_model(model_s)
                model_t = get_model(model_t)
                Ki.append(
                    sill * model_s(dist_s, 1., param_s) * model_t(dist_t, 1., param_t))
            Ki_sum = sum(Ki)
            return Ki_sum, Ki  # K, KK in matlab
        else:
            (modelS,) = model_res
            dist_s = coord2dist(c1[:, 0:2], c2[:, 0:2])
            dist_t = coord2dist(c1[:, 2:3], c2[:, 2:3])
            Ki = []
            for model_s, param_i in zip(modelS, params):
                sill, param_s, s_t_ratio = param_i
                model_s = get_model(model_s)
                Ki.append(
                    sill * model_s(dist_s + s_t_ratio * dist_t, 1., param_s))
            Ki_sum = sum(Ki)
            return Ki_sum, Ki  # K, KK in matlab
    else:
        Ki = []
        dist_s = coord2dist(c1, c2)
        (modelS,) = model_res
        for model_s, param_i in zip(modelS, params):
            sill, param_s = param_i
            model_s = get_model(model_s)
            Ki.append(sill * model_s(dist_s, 1., param_s))
        Ki_sum = sum(Ki)
        return Ki_sum, Ki  # K, KK in matlab

def coord2Ksplit(c1_split, c2split, models, params):
    '''
    split dataset for estimated/hard/soft data split.
    
    c1_split  list                 list of m 2D numpy array of S/T coordinates. 
                                   Each component of list can be an arbitary m by nd
                                   array with coordinates. 
    c2_split  list                 list of n 2D numpy array of S/T coordinates    

    models    list of string       covmodels, a sequence contains covariance models string
    params    list of 
              sequence of float    covparams, a list contains a sequence of covariance parameters values


    return 
    sumK      2D list              m by n 2D list of covariances between the 
                                   elements of c1_split and c2_split  
    Ki        list of              m by n 2D list of nested covariances between the 
              i_th model value     elements of c1_split and c2_split. Each nested
                                   covariance has K components stored in a list
    
    Note: 
    Let c1=[c1a,c1b,c1c] and c2=[c2a,c2b,c2c]  
    in the 2D list, the estimated covariance are expressed as below
    [[[c1a,c2a],[c1a,c2b],[c1a,c2c]],
     [[c2a,c2a],[c2a,c2b],[c2a,c2c]],
     [[c3a,c2a],[c3a,c2b],[c3a,c2c]]]    
     
    if any elements in c1 or c2 are None, their covariances are specified by None  
    
    Remark: 
    covariance format
    isST
        isSTsep
            models: ['exponentialC','exponentialC','...'] 
            params: [(3,None), (21.9, 35.8)] 
        not isSTsep
            models: ["gaussianCST", "exponentialCST", "..." ]
            params: [(sill1, bs1, stratio),
                     (sill2, bs2, stratio)]
    not isST
        models: ["gaussian", "exponential", "..."" ]
        params: [(sill1, bs1),
                 (sill2, bs2)]

    '''
    sum_k_split = []
    ki_split = []
    for c1_i in c1_split:
        sum_k_j = []
        ki_split_j = []
        for c2_j in c2split:
            if c1_i is not None and c2_j is not None:
                sum_k, ki = coord2K(c1_i, c2_j, models, params)
                sum_k_j.append(sum_k)
                ki_split_j.append(ki)
            else:
                sum_k_j.append(None)
                ki_split_j.append(None)
        sum_k_split.append(sum_k_j)
        ki_split.append(ki_split_j)
        
    return sum_k_split, ki_split

def coord2Kcombine(sum_k_split):
    '''
    conbine coord2K split result, back to coord2K
    '''
    output = []
    for i in sum_k_split:
        r = [j for j in i if j is not None]
        if r:
            output.append(np.hstack(r))
    if output:
        output = np.vstack(output)
    return output
