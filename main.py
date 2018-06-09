#-----------------------------------------------------------------------------------------------------------------------
#@创建于2017-9
#@Author：lieying
#@School:USTB
#@E-mil:kwj123321@163.com
#----
import pandas as pd
import numpy as np
import get_feat_XGBmodel as feat_model
import Model_confuse_sub_result as M_C_S_res
import model_lgb as lgbxgb
#===============读数据===============

feat_3,train_3 = feat_model.read_file('feature_3','train_3',0)
feat_4,train_4 = feat_model.read_file('feature_4','train_4',0)
feat_5,train_5 = feat_model.read_file('feature_5','train_5',0)
feat_6,train_6 = feat_model.read_file('feature_6','train_6',0)
feat_7_p,train_7_p = feat_model.read_file('feature_7_2016','train_7_2016',0)
feat_7,train_7 = feat_model.read_file('feature_7_seg2','train_7_seg2',0)
#===============数据链接===============
print("正在连接数据....")
#-特征
feat_3_4_5 = pd.concat([feat_3,feat_4,feat_5], axis=0)
feat_4_5_6 = pd.concat([feat_4,feat_5,feat_6], axis=0)
#feat_3_4_5_p7  = pd.concat([feat_3,feat_4,feat_5,feat_7], axis=0)
feat_3_4_5_6_p7  = pd.concat([feat_3,feat_4,feat_5,feat_6,feat_7_p], axis=0)
#-训练集
#train_3_4_5_p7 = pd.concat([train_3,train_4,train_5,train_7],axis =0)
train_3_4_5 = pd.concat([train_3,train_4,train_5],axis =0)
train_4_5_6 = pd.concat([train_4,train_5,train_6],axis =0)
train_3_4_5_6_p7 = pd.concat([train_3,train_4,train_5,train_6,train_7_p],axis =0)

print("完成！",train_3_4_5_6_p7.shape)
#====================================================模型==========================================================
#--训练集提取特征
train_3_4_5 = feat_model.get_feat(feat_3_4_5,train_3_4_5)
train_3_4_5.to_csv('data/train_3_4_5_before_spilt_point.txt', index=False)
train_4_5_6 = feat_model.get_feat(feat_4_5_6,train_4_5_6)
train_4_5_6.to_csv('data/train_4_5_6_before_spilt_point.txt', index=False)

#train_3_4_5_p7 = feat_model.get_feat(feat_3_4_5_p7,train_3_4_5_p7)
train_3_4_5_6_p7 = feat_model.get_feat(feat_3_4_5_6_p7,train_3_4_5_6_p7)

train_6 = feat_model.get_feat(feat_6,train_6)
train_6.to_csv('data/train_6_end.txt',index=False)
train_7 = feat_model.get_feat(feat_7,train_7)
train_7.to_csv('data/train_7_end.txt',index=False)
#=============================================划分训练集================================================================
#====================================================模型==============================================================
#----横向拼接
train_456_345_fuse = feat_model.confuse_col('train_4_5_6_before_spilt_point','train_3_4_5_before_spilt_point',1)
train_456_345_fuse.to_csv('data/train_456_345_fuse.txt',index=False)

train_7_6_fuse = feat_model.confuse_col('train_7_end','train_6_end',1)
train_7_6_fuse.to_csv('data/train_7_6_fuse_end.txt',index=False)

#-训练集去噪，并存文件
train_456_345_fuse = feat_model.To_file_remove_noise_point(train_456_345_fuse)
train_456_345_fuse.to_csv('data/train_456_345_fuse_end.txt', index=False)

#train_4_5_6 = feat_model.To_file_remove_noise_point(train_4_5_6)
#train_4_5_6.to_csv('data/train_4_5_6_end.txt', index=False)
#train_3_4_5_p7 = feat_model.To_file_remove_noise_point(train_3_4_5_p7)
#train_3_4_5_p7.to_csv('data/train_3_4_5_p7.txt', index=False)
train_3_4_5_6_p7 = feat_model.To_file_remove_noise_point(train_3_4_5_6_p7)
train_3_4_5_6_p7.to_csv('data/train_3_4_5_6_p7_end.txt', index=False)
#=========================================================训练==========================================================

#------------线上七月--------
#-----需要注意文件名-------
#--模型1
#xgb.XGB_train('train_456_345_fuse','train_7_6_fuse_end',"on",1)
#--模型2
#xgb.XGB_train('train_3_4_5_6_p7','train_7_end',"on",1)
#--模型3
#lgb.LGBM_train('train_456_345_fuse','train_7_6_fuse_end',915,"on")
#--模型4
#lgb.LGBM_train('train_3_4_5_6_p7_end','train_7_end',915,"on")
#--模型融合
M_C_S_res.offline_confuse("on")
print("================succeed!!!=============")
#----------------------lgb---------------












