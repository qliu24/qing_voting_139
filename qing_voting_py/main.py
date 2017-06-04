from config_voting import *
from merge_det import merge_det
from evalVotingModelForObjDet import evalVotingModelForObjDet

# merge_det(all_categories+all_bgs, all_categories)
exec(open("./compt_all_score_nms_list.py").read())
for oo in all_categories:
    print(oo)
    evalVotingModelForObjDet(oo,oo)
