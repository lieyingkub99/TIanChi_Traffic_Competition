# TIanChi_Traffic_Competition
阿里天池智慧交通预测挑战赛

#@创建于2017-9
#@Author：lieying
#@School:USTB
#@E-mail:kwj123321@163.com

复赛：Top7 /1716队

#----
此算法为智慧交通预测挑战赛复赛程序

1、先运行sub_handle.py生成提交样本，然后运行link_top_process.py生成道路基本信息。
（@@@最重要的的是get_feat_XGBmodel.py这个文件用于提取特征，写了相应的函数，直接调用。）

2、先运行get_feat.py进行划分数据集

3、运行get_feat_2016_7.py和get_feat_2017_3.py进行提取2016年的七月和2017年的3月

4、然后运行main.py进行提取和去除噪点。然后送入XGBoost和lightgbm模型进行训练得到四个模型的结果，然后融合处理。



