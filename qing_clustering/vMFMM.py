import numpy as np

def normalize_features(features):
    '''features is N by d matrix'''
    assert(len(features.shape)==2)
    return features/np.sqrt(np.sum(features**2, axis=1).reshape(-1,1))

class vMFMM:
    def __init__(self, cls_num, init_method = 'random', max_it):
        self.cls_num = cls_num
        self.init_method = init_method
        
        
    def fit(self, features, normalized = False):
        self.features = features
        if not normalized:
            self.features = normalize_features(features)
            
        self.d = self.features.shape[1]
        self.kappa = np.sqrt(self.d/2)  # all clusters share the same concentration parameter
        self.pi = np.ones(self.cls_num)/self.cls_num
        if init_method=='random':
            self.mu = np.random.random((self.cls_num, self.d))
            self.mu = normalize_features(self.mu)
            
    def e_step(self):
        self.p=
        
    def m_step(self):
        self.pi
        self.mu
        self.r
        self.kappa
        
        
    