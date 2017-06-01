from config_voting import *

cluster_num = featDim
cluster_file = '/home/candy/qing_voting_139/qing_voting_py/data/dictionary_PASCAL3D+_VGG16_{0}_K{1}.pickle'.format(VC['layer'], cluster_num)
save_path1 = '/home/candy/qing_voting_139/qing_voting_py/data/dictionary_PASCAL3D+_VGG16_{0}_K'.format(VC['layer'])
save_path2 = '_prune_{0}.pickle'.format(cluster_num)

print('loading data...')

# number of files to read in
file_num = 6
feat_set = np.zeros((featDim, 0))
for ii in range(file_num):
    print('loading file {0}/{1}'.format(ii+1, file_num))
    fname = Dict['cache_path']+str(ii)+'.pickle'
    with open(fname, 'rb') as fh:
        res, _, _ = pickle.load(fh)
        feat_set = np.column_stack((feat_set, res))
        

print('all feat_set')
print(feat_set.shape)

with open(cluster_file, 'rb') as fh:
    assignment, centers = pickle.load(fh)
    
print('pruning the clusters...')

# L2 normalization
feat_norm = np.sqrt(np.sum(feat_set**2, 0))
feat_set = feat_set/feat_norm


print('compute metric...')
# decide the rank of clusters
K = cluster_num
count = np.bincount(assignment, minlength=K)

# based on centers
pw_cen = np.zeros((K,K))
for k in range(K):
    for m in range(K):
        pw_cen[k,m] = np.linalg.norm(centers[k]-centers[m])


# based on data points
pw_all = np.zeros((K,K))
for k in range(K):
    target = centers[k]
    for m in range(K):
        index = np.where(assignment==m)[0]
        temp_feat = feat_set[:, index]
        dist = np.sqrt(np.sum((temp_feat - target.reshape(-1,1))**2, axis=0))
        sort_value = np.sort(dist)
        pw_all[k, m] = np.mean(sort_value[0:int(0.95*len(sort_value))])


list = np.zeros(K)
for k in range(K):
    rec = np.zeros(K)
    for m in range(K):
        if m != k:
            rec[m] = (pw_all[m,m] + pw_all[k,k])/pw_cen[m,k]
        
    list[k] = np.max(rec)


# the lower the better
bbb = np.argsort(list)
aaa = np.sort(list)
sort_list = np.stack([aaa,bbb])

# the higher the better
count_norm = count/np.sum(count)
bbb = np.argsort(count_norm)[::-1]
aaa = np.sort(count_norm)[::-1]
sort_count_norm = np.stack([aaa,bbb])

# give big penalty if cluster number is too small
penalty = 100*(count<100)

# combine the above metrics, the lower the better
com = list - K*count_norm + penalty
bbb = np.argsort(com)
aaa = np.sort(com)
sort_com = np.stack([aaa,bbb])


print('greedy pruning...')
sort_cls = sort_com[1].astype(int)
rec = np.ones(K)
thresh1 = 0.95  # magic number
thresh2 = 0.2  # magic number
prune = np.zeros((3,0))
prune_res = []

while np.sum(rec) > 0:
    temp = np.zeros((3,0))
    idx = np.where(rec==1)[0][0]
    cls = sort_cls[idx]
    
    target = centers[cls]
    index = np.where(assignment==cls)[0]
    temp_feat = feat_set[:, index]
    dist = np.sqrt(np.sum((temp_feat - target.reshape(-1,1))**2, axis=0))
    sort_value = np.sort(dist)
    dist_thresh = sort_value[int(thresh1*len(sort_value))]
    
    rec[idx] = 0
    for n in range(idx+1,K):
        if rec[n] == 1:
            index = np.where(assignment==sort_cls[n])[0]
            temp_feat = feat_set[:, index]
            dist = np.sqrt(np.sum((temp_feat - target.reshape(-1,1))**2, axis=0))
            if np.mean(dist<dist_thresh) >= thresh2:
                temp = np.column_stack([temp, np.array([n,sort_cls[n],np.mean(dist<dist_thresh)]).reshape(-1,1)])
                rec[n] = 0
                
                
    print('{0}, {1}, {2}'.format(idx, cls, temp.shape[1]))
    prune = np.column_stack([prune, np.array([idx, cls, temp.shape[1]]).reshape(-1,1)])
    prune_res.append(temp)


print('update new dictionary...')
K_new = prune.shape[1]
print('new K is : {0}'.format(K_new))

pruning_table=[None for nn in range(K_new)]

centers_new = np.zeros((K_new, centers.shape[1]))
assignment_new = np.zeros_like(assignment)
for k in range(K_new):
    if prune_res[k].size == 0:
        temp = np.array([prune[1,k]])
    else:
        temp = np.append(prune[1,k], prune_res[k][1,:])
    
    temp=temp.astype(int)
    pruning_table[k]=temp
    weight = count[temp]
    weight = weight/sum(weight)
    temp_cen = centers[temp]
    centers_new[k] = np.dot(temp_cen.T, weight.reshape(-1,1)).squeeze()
    for i in range(len(temp)):
        assignment_new[assignment==temp[i]] = k


K = K_new
centers = centers_new
assignment = assignment_new

save_path = save_path1+str(K)+save_path2
with open(save_path, 'wb') as fh:
    pickle.dump([assignment, centers], fh)
