import numpy as np
from scipy.misc import logsumexp

def normalize_features(features):
    '''features is N by d matrix'''
    assert(len(features.shape)==2)
    return features/np.sqrt(np.sum(features**2, axis=1).reshape(-1,1))

class vMFMM_hard:
    def __init__(self, cls_num, init_method = 'random'):
        self.cls_num = cls_num
        self.init_method = init_method
        
        
    def fit(self, features, kappa, max_it, normalized = False, verbose=True):
        self.features = features
        if not normalized:
            self.features = normalize_features(features)
            
        self.n, self.d = self.features.shape
        # self.kappa = np.sqrt(self.d/2)  # all clusters share the same concentration parameter
        self.kappa2 = kappa
        
        self.pi = np.random.random(self.cls_num)
        self.pi /= np.sum(self.pi)
        if self.init_method =='random':
            self.mu = np.random.random((self.cls_num, self.d))
            # self.mu = np.array([[1,0],[0,1]])
            self.mu = normalize_features(self.mu)
        elif self.init_method =='k++':
            print('start k++')
            centers = []
            centers_i = []
            cos_dis = 1-np.dot(self.features, self.features.T)
            print('finish cos_dis')
            centers_i.append(np.random.choice(self.n))
            centers.append(self.features[centers_i[0]])
            for i in range(self.cls_num-1):
                if i%10==0:
                    print('k++ center {0}'.format(i))
                    
                subset_idx = np.random.choice(self.n, size=(100,), replace=False)
                
                prob = np.min(cos_dis[subset_idx,:][:,centers_i], axis=1)**2
                prob /= np.sum(prob)
                centers_i.append(np.random.choice(subset_idx, p=prob))
                centers.append(self.features[centers_i[-1]])
                
            self.mu = np.array(centers)
            print('finish k++')
            
        for itt in range(max_it):
            self.e_step()
            self.m_step()
            if verbose and itt%1==0:
                print("iter {0}: {1},{2}".format(itt, self.mllk, self.kappa))
            
            
    def e_step(self):
        # update p
        logP = np.dot(self.features, self.mu.T)*self.kappa2 + np.log(self.pi).reshape(1,-1)  # n by k
        # logP_norm = logP - logsumexp(logP, axis=1).reshape(-1,1)
        # self.p = np.exp(logP_norm)
        q = np.argmax(logP, axis=1)
        self.p = np.zeros((self.n, self.cls_num))
        for nn in range(self.n):
            self.p[nn,q[nn]] = 1
            
        self.mllk = np.mean(logsumexp(logP, axis=1))
        
        
    def m_step(self):
        # update pi and mu
        self.pi = np.sum(self.p, axis=0)/self.n
        
        self.mu = np.sum(np.tile(self.features.reshape(self.n,1,self.d),(1,self.cls_num,1))*self.p.reshape(self.n,self.cls_num,1),axis=0)/np.sum(self.p, axis=0).reshape(-1,1)
        
        # for cc in range(self.cls_num):
        #     self.mu[cc] = np.sum(self.p[:,cc].reshape(-1,1) * self.features, axis=0)/np.sum(self.p[:,cc])
        
        r = np.mean(np.sqrt(np.sum(self.mu**2, axis=1))*self.pi)
        # r = np.mean(np.sqrt(np.sum(self.mu**2, axis=1))/(self.n*self.pi))
        
        self.kappa = (r*self.d-r**3)/(1-r**2)
            
        self.mu = normalize_features(self.mu)
        
        
        
        
    