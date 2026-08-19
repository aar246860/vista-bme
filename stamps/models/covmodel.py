# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import gamma, kv

def get_model(m_name):
    'use model name to get model'
    try:
        return eval(m_name)
    except NameError as e:
        #must be do something...
        raise e

def exponentialC(dist, sill, ar, jac=False, jacpar=None):
    '''
    Exponential covariance function

    Input:
    dist      m by n    an 2D array of spatial or S/T distances
    sill      scalar    covariance sill
    ar        scalar    covariance correlation length
    jac       bool      if the return should be jacobian of parameters.
                        the default is False to return the covariance estimation
    jacpar    string    To determine which parameter is used for Jacobian
                        estimation, the default is None.
                        Options are 'sill' and 'ar'
    
    Output:
    cov/jac   m by n    an 2D array of covariance or Jacobian estimation 
                        cooresponding to dist
    '''

    if not jac:
        cov = sill * np.exp(-3 * dist / ar)
        return cov
    else:
        if jacpar == 'sill':
            jac = np.exp(-3 * dist / ar)
        elif jacpar == 'ar':
            jac = sill*3*dist/(ar**2)*np.exp(-3*dist/ar)
        elif jacpar == 'ar2':
            jac = sill*((3*dist/(ar**2))**2-6*dist/(ar**3))*np.exp(-3*dist/ar)      
        return jac

def gaussianC(dist, sill, ar, jac=False, jacpar=None):
    if not jac:
        return sill * np.exp(-3 * (dist / ar)**2)
    else:
        if jacpar == 'sill':
            jac = np.exp(-3 * (dist / ar)**2)
        elif jacpar == 'ar':
            jac = sill * np.exp(-3 * (dist / ar)**2)*(6*(dist**2)/ar**3)
        elif jacpar == 'ar2':
            jac = sill * np.exp(-3*(dist/ar)**2) *\
                ((6*(dist**2)/ar**3)**2 - 18*(dist**2)/(ar**4))
        return jac

def sphericalC(dist, sill, ar, jac=False, jacpar=None):
    value = dist / ar
    if not jac:    
        result = sill * (1- ((3/2.) * value - (1/2.) * (value)**3))
        result[dist > ar] = 0.
        return result
    else:
        if jacpar == 'sill':
            jac = 1 - ((3/2.) * value - (1/2.) * (value)**3)
            jac[dist > ar] = 0.
        elif jacpar == 'ar':
            jac = sill*(3/2.*dist/ar**2-3/2.*dist**3/ar**4)   
            jac[dist > ar] = 0.
        elif jacpar == 'ar':
            jac = sill*(-3.*dist/ar**3-6.*dist**3/ar**5)   
            jac[dist > ar] = 0.      
        return jac  
    
def holecosC(dist, a_half, p_half):
    return a_half * np.cos(np.pi * dist / p_half)

def nuggetC(dist, sill, ar=None, jac=False, jacpar=None):
    if not jac:
        result = np.zeros(dist.shape)
        result[dist == 0] = sill
        return result
    else:
        jac=np.zeros([0,0])
        return jac

def maternC(dist, sill, parms):
    '''
    Matern covariance with practical distance adjustment
    
    C = maternC(dist, sill, parms)
    
    Input
    dist    m by n     an numpy array of S/T distances
    sill    scalar     sill
    parms   list       a list of two components 
                       (correlation length, shape parameter).
                       
    Output
    Cov     m by n     an numpy array of S/T covariances                   
    '''

    ar = parms[0]
    shp = parms[1]
    scale = np.sqrt(2*shp*6)*(3./2)**(1./4./shp)
    sdist = scale*dist
    result =\
        (sill/(2.**(shp-1))/gamma(shp)) * ((sdist/ar)**shp) * kv(shp,sdist/ar)
    result[np.where(sdist==0)] = sill
    return result
