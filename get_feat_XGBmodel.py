#-----------------------------------------------------------------------------------------------------------------------
#@创建于2017-9
#@Author：lieying
#@School:USTB
#@E-mil:kwj123321@163.com
#----
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
#================函数==================
def mape_object(y,d):

    g=1.0*np.sign(y-d)/d
    h=1.0/d
    return -g,h

# 评价函数
def mape(y,d):
    c=d.get_label()
    result=np.sum(np.abs(y-c)/c)/len(c)
    return "mape",result

# 评价函数ln形式
def mape_ln(y,d):
    c=d.get_label()
    result=np.sum(np.abs(np.expm1(y)-np.abs(np.expm1(c)))/np.abs(np.expm1(c)))/len(c)
    return "mape",result
from scipy.stats import mode
# 中位数
def mode_function(df):
    counts = mode(df.round(0))
    return counts[0][0]

#=========================================读数据并，训练集组合函数======================================================
#==读数据
def read_file(i,j,p):
    print("---正在读取数据...")
    feat_data = pd.read_csv('data/%s.txt'% (i),low_memory=False)
    train= pd.read_csv('data/%s.txt'% (j), low_memory=False)
    if p != 0:
        print("============%d 月feat_data的基本信息=======" %(p))
        print(feat_data.info())
        print("============%d 月train的基本信息=======" % (p))
        print(train.info())
    print("---读取数据完成！")
    return feat_data,train

#=============================================特征提取函数==============================================================
def get_feat(feat,train):
    print("=======正在特征提取=====")
    feature_6_13_16 = feat.loc[(feat.time_interval_begin_hour == 6) | (feat.time_interval_begin_hour == 13) | (feat.time_interval_begin_hour == 16)]
    feature_7_14_17 = feat.loc[(feat.time_interval_begin_hour == 7) | (feat.time_interval_begin_hour == 14) | (feat.time_interval_begin_hour == 17)]
    # --按（ID,month，day,hour）分
    for i in [58,48,38,28,18,8]:
        print("---------:  ",i)
        # --7点
        tmp_1 = feature_7_14_17.loc[(feature_7_14_17.time_interval_minutes >= i), :]
        tmp_1 = tmp_1.groupby(['link_ID', 'time_interval_month', 'time_interval_day', 'time_interval_begin_hour'])[
            'travel_time'].agg([('median_1_%d' % (i), np.median), ('mode_1_%d' % (i), mode_function)]).reset_index()
        tmp_1["time_interval_begin_hour"] = tmp_1.time_interval_begin_hour + 1
        train = pd.merge(train, tmp_1,on=['link_ID', 'time_interval_month', 'time_interval_day', 'time_interval_begin_hour'], how='left')
        # --6点
        tmp_2 = feature_6_13_16.loc[(feature_6_13_16.time_interval_minutes >= i), :]
        tmp_2 = tmp_2.groupby(['link_ID', 'time_interval_month', 'time_interval_day', 'time_interval_begin_hour'])[
            'travel_time'].agg([('median_2_%d' % (i), np.median), ('mode_2_%d' % (i), mode_function)]).reset_index()
        tmp_2["time_interval_begin_hour"] = tmp_2.time_interval_begin_hour + 2
        train = pd.merge(train, tmp_2,on=['link_ID', 'time_interval_month', 'time_interval_day', 'time_interval_begin_hour'],  how='left')
    #--按（ID,month，day）分
    tmp_3 = feat.groupby(['link_ID', 'time_interval_month', 'time_interval_day'])['travel_time'].agg([('median_3', np.median), ('mode_3', mode_function)]).reset_index()
    train = pd.merge(train, tmp_3, on=['link_ID', 'time_interval_month', 'time_interval_day'], how='left')
    #--按（ID,month，day，minutes）分
    train_month_day_minute =feat.groupby(['link_ID', 'time_interval_month', 'time_interval_day', 'time_interval_minutes'])[ 'travel_time'].agg([('median_month_day_minute', np.median)]).reset_index()
    train = pd.merge(train, train_month_day_minute, on=['link_ID', 'time_interval_month', 'time_interval_day', 'time_interval_minutes'],  how='left')
    #--将6点直接megre到8点
    tmp_4 = feature_6_13_16[['link_ID', 'time_interval_month', 'time_interval_day', 'time_interval_begin_hour', 'time_interval_minutes', 'travel_time']]
    tmp_4["time_interval_begin_hour"] = tmp_4.time_interval_begin_hour + 2
    tmp_4 = tmp_4.rename(columns={'travel_time': 'travel_time_6'})
    train = pd.merge(train, tmp_4, on=['link_ID', 'time_interval_month', 'time_interval_day', 'time_interval_begin_hour','time_interval_minutes'], how='left')
    # 将7点直接megre到8点
    tmp_5 = feature_7_14_17[['link_ID', 'time_interval_month', 'time_interval_day', 'time_interval_begin_hour', 'time_interval_minutes',  'travel_time']]
    tmp_5["time_interval_begin_hour"] = tmp_5.time_interval_begin_hour + 1
    tmp_5 = tmp_5.rename(columns={'travel_time': 'travel_time_7'})
    train = pd.merge(train, tmp_5,on=['link_ID', 'time_interval_month', 'time_interval_day', 'time_interval_begin_hour', 'time_interval_minutes'], how='left')
    # --按（ID,month,minutes）分
   # --按（ID,minutes）分
    train_median_minute = feat.groupby(['link_ID', 'time_interval_minutes'])['travel_time'].agg( [('median_minute', np.median)]).reset_index()
    train = pd.merge(train, train_median_minute, on=['link_ID', 'time_interval_minutes'],how='left')
    #--按(ID，week，minutes)分
    train_median_minute = feat.groupby(['link_ID', 'time_interval_month', 'time_interval_week','time_interval_minutes'])['travel_time'].agg( [('medi_mon_week_minute', np.median)]).reset_index()
    train = pd.merge(train, train_median_minute, on=['link_ID', 'time_interval_month','time_interval_week', 'time_interval_minutes'], how='left')
    print("=======特征提取完成行列数:  ",train.shape)
    return train
#================================================去除噪点===============================================================
def To_file_remove_noise_point(train):
    print("=======正在去除噪点=====")
    print("去除噪点以前的行列数：", train.shape)
    train_tmp =train.groupby(['link_ID', 'time_interval_month', 'time_interval_day', 'time_interval_begin_hour'])[ 'travel_time'].median().reset_index().rename(columns={'travel_time': 'median_'})
    train = pd.merge(train, train_tmp, on=['link_ID', 'time_interval_month', 'time_interval_day', 'time_interval_begin_hour'],how='left')
    train = train.loc[(train.travel_time) <= (1.45* (train.median_))]
    print("去除噪点以后的行列数：", train.shape)
    return train
#================================================两个模型融合===========================================================
def confuse_col(train_this,train_last,i):
    print("正在拼接...")
    train_this = pd.read_csv('data/%s.txt' % (train_this), low_memory=False)
    train_last = pd.read_csv('data/%s.txt' % (train_last), low_memory=False)
    string = ['time_interval_year','time_interval_week','time_interval_begin','length', 'width', 'link_class', 'week_1', 'week_2', 'week_3', 'week_4', 'week_5', 'week_6', 'week_7', 'i_num', 'o_num',
       'in_first', 'in_second', 'in_third', 'in_forth', 'out_first', 'out_second', 'out_third', 'out_forth', 'length_in1', 'width_in1',
       'link_class_in1', 'length_in2', 'width_in2', 'link_class_in2', 'length_in3', 'width_in3', 'link_class_in3', 'length_in4', 'width_in4',
       'link_class_in4', 'length_out1', 'width_out1', 'link_class_out1',  'length_out2', 'width_out2', 'link_class_out2', 'length_out3',
       'width_out3', 'link_class_out3', 'length_out4', 'width_out4', 'link_class_out4','travel_time']
    if i==1:
        train_last = train_last.drop(string, axis=1)
        train_last["time_interval_month"] = train_last.time_interval_month+1
        train = pd.merge(train_this, train_last, on=['link_ID', 'time_interval_month', 'time_interval_day', 'time_interval_begin_hour','time_interval_minutes'], how='left')
        print("拼接完成！")
        return train
    if i==2:
        train = pd.concat([train_last,train_last], axis=0)
        print("拼接完成！")
        return train
#======================================================训练============================================================
def XGB_train(t1,t2,on_or_off,i,mode):
    train = pd.read_csv('data/%s.txt'%(t1),low_memory=False)
    train_label = np.log1p(train.pop('travel_time'))

    test = pd.read_csv('data/%s.txt'%(t2),low_memory=False)
    test_lable =  np.log1p(test.pop('travel_time'))

    train.drop(['time_interval_begin','time_interval_week','median_'],inplace=True,axis=1)
    test.drop(['time_interval_begin','time_interval_week'],inplace=True,axis=1)
    print("训练样本和测试样本行列数")
    print(train.shape,test.shape)

    if mode == 1:
        xlf = xgb.XGBRegressor(max_depth=11,
                               learning_rate=0.01,
                               n_estimators=425,
                               silent=True,
                               objective=mape_object,
                               gamma=0,
                               min_child_weight=6,
                               max_delta_step=0,
                               subsample=0.8,
                               colsample_bytree=0.8,
                               colsample_bylevel=1,
                               reg_alpha=1e0,
                               reg_lambda=0,
                               scale_pos_weight=1,
                               seed=12,
                               missing=None)
        xlf.fit(train.values, train_label.values, eval_metric=mape_ln, verbose=True, eval_set=[(test.values, test_lable.values)])
        print("正在预测...")
        result = xlf.predict(test.values)
    elif mode ==2:
        clf = RandomForestRegressor(n_estimators=20,
                                    max_depth=9,
                                    min_samples_split=3,
                                    min_samples_leaf=5,
                                    n_jobs=-1,
                                    random_state=0)
        clf.fit(train.values, train_label.values)
        result = clf.predict(test.values)
    if on_or_off == "off":
        print("写入文件...")
        #写入文件
        test["travel_time"] = test_lable
        test['travel_time'] = np.round(np.expm1(test['travel_time']), 6)
        test['travel_time_%d'%(i)] = result
        test['travel_time_%d' % (i)] = np.round(np.expm1(test['travel_time_%d'%(i)]), 6)
        test[['link_ID', 'time_interval_day', 'time_interval_begin_hour', 'time_interval_minutes','travel_time',
              'travel_time_%d'%(i)]].to_csv('sub_data/model_%d_offline_XGB.txt'%(i), index=False)
    if on_or_off == "on":
        print("正在预测...")
        result = xlf.predict(test.values)
        print("写入文件...")
        test['travel_time'] = result
        test.travel_time = np.round(np.expm1(test.travel_time), 6)
        gy_teample_sub = pd.read_csv('pre_data/gy_teample_sub_seg2.txt', low_memory=False)
        test = test[['link_ID', 'time_interval_day', 'time_interval_begin_hour', 'time_interval_minutes', 'travel_time']]
        gy_teample_sub = pd.merge(gy_teample_sub, test,on=['link_ID', 'time_interval_day', 'time_interval_begin_hour','time_interval_minutes'], how="left")
        gy_teample_sub[['link_ID', 'date_time', 'time_interval', 'travel_time']].to_csv('sub_data/Fighting666666_0915_2.txt', sep='#', index=False, header=False)
        print(gy_teample_sub[['link_ID', 'date_time', 'time_interval', 'travel_time']].shape)
        print(gy_teample_sub[['link_ID', 'date_time', 'time_interval', 'travel_time']].isnull().sum())
        print("====================succeed!=================")













