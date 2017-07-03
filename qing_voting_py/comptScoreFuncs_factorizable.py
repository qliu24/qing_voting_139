import math
import numpy as np
from scipy.misc import logsumexp

def dist_to_encode(dist, stride=2, magic_thh=0.85):
    '''
    ihh, iww, idd = dist.shape
    
    tmp = np.zeros((math.ceil(ihh/stride), math.ceil(iww/stride), idd)).astype('int')
    for hh in range(0,ihh,stride):
        for ww in range(0,iww,stride):
            d_min = np.unravel_index(np.argmin(dist[hh:hh+stride,ww:ww+stride,:]), \
                                 dist[hh:hh+stride,ww:ww+stride,:].shape)
            if np.min(dist[hh:hh+stride,ww:ww+stride,:]) < magic_thh:
                tmp[int(hh/stride),int(ww/stride),d_min[2]] = 1
    
    
    tmp = np.ones((math.ceil(ihh/stride), math.ceil(iww/stride), idd))
    for dd in range(idd):
        for hh in range(0,ihh,stride):
            for ww in range(0,iww,stride):
                min_dist = np.min(dist[hh:hh+stride, ww:ww+stride, dd])
                tmp[int(hh/stride),int(ww/stride),dd] = min_dist
    '''
    return (dist<magic_thh).astype(int)


def comptScores(dist, obj_weight, logZ, if_encode = False):
    if not if_encode:
        inp = dist_to_encode(dist)
    else:
        inp = dist
        
    hi,wi,ci = inp.shape
    ho,wo,co = obj_weight.shape
    assert(ci == co)
    if hi > ho:
        diff1 = math.floor((hi-ho)/2)
        inp = inp[diff1: diff1+ho, :, :]
    else:
        diff1 = math.floor((ho-hi)/2)
        diff2 = ho-hi-diff1
        inp = np.pad(inp, ((diff1, diff2),(0,0),(0,0)), 'constant', constant_values=0)
        
    assert(inp.shape[0] == ho)
    
    if wi > wo:
        diff1 = math.floor((wi-wo)/2)
        inp = inp[:, diff1: diff1+wo, :]
    else:
        diff1 = math.floor((wo-wi)/2)
        diff2 = wo-wi-diff1
        inp = np.pad(inp, ((0,0),(diff1, diff2),(0,0)), 'constant', constant_values=0)
        
    assert(inp.shape[1] == wo)
    
    term1 = inp*obj_weight
    score = np.sum(term1) - logZ
    return score
    
    
def comptScoresM(dist, obj_weights, logZs, log_priors):
    inp = dist_to_encode(dist)
    K = len(obj_weights)
    score_i = np.zeros(K)
    for kk in range(K):
        logllk = comptScores(inp, obj_weights[kk], logZs[kk], if_encode=True)
        score_i[kk] = logllk + log_priors[kk]
    
    score = logsumexp(score_i)
    return score