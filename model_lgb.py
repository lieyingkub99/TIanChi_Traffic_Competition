#-----------------------------------------------------------------------------------------------------------------------
#@创建于2017-9
#@Author：lieying
#@School:USTB
#@E-mil:kwj123321@163.com
#----
import pandas as pd
import numpy as np
import time
import math
#====
#off:0.2762   on:0.2625
#off:0.2758   on:0.26208
#=
import lightgbm as lgb
def mape_object(y, d):
    y = (y)
    gaps = (d.get_label())
    grad = np.sign(y-gaps)/gaps
    hess = 1/gaps
    grad[(gaps == 0)] = 0
    hess[(gaps == 0)] = 0
    return grad,hess

# 评价函数ln形式
def mape_ln(y, d):
    c = d.get_label()
    result = np.sum(np.abs(np.expm1(y)-np.abs(np.expm1(c)))/np.abs(np.expm1(c)))/len(c)

    return "mape", result, False
def LGBM_train(t1,t2,i,on_off):

    train = pd.read_csv('data/%s.txt'%(t1),low_memory=False)
    train_label = np.log1p(train.pop('travel_time'))

    test = pd.read_csv('data/%s.txt'%(t2),low_memory=False)
    test_lable =  np.log1p(test.pop('travel_time'))
    train.drop(['link_ID','in_first', 'in_second', 'in_third','in_forth','out_first', 'out_second','out_third','out_forth','time_interval_begin','time_interval_week','median_'],inplace=True,axis=1)
    test.drop(['link_ID','in_first', 'in_second', 'in_third','in_forth','out_first', 'out_second','out_third','out_forth','time_interval_begin','time_interval_week',],inplace=True,axis=1)
    print("训练样本和测试样本行列数")
    print(train.shape,test.shape)



    if on_off == "off":
        print("正在预测...")
        result = runLGBM(train, train_label, test, test_lable,on_off)
        print("写入文件...")
        # 写入文件
        test = pd.read_csv('data/%s.txt' % (t2), low_memory=False)
        test = test[
            ['link_ID', 'time_interval_day', 'time_interval_begin_hour', 'time_interval_minutes', 'travel_time']]
        test["travel_time"] = test_lable
        test['travel_time'] = np.round(np.expm1(test['travel_time']), 6)
        test['travel_time_%d' % (i)] = result
        test['travel_time_%d' % (i)] = np.round(np.expm1(test['travel_time_%d' % (i)]), 6)
        test.to_csv('sub_data/model_%d_offline.txt' % (i), index=False)

        print("完成...")
    if on_off == "on":

        print("正在训练预测...")
        result = runLGBM(train, train_label, test, test_lable,on_off)
        print("写入文件...")
        test_result = pd.read_csv('data/%s.txt' % (t2), low_memory=False)
        test_result.drop(["travel_time"],inplace=True,axis=1)
        test_result['travel_time'] = result
        test_result.travel_time = np.round(np.expm1(test_result.travel_time), 6)
        gy_teample_sub = pd.read_csv('pre_data/gy_teample_sub_seg2.txt', low_memory=False)
        test_result = test_result[
            ['link_ID', 'time_interval_day', 'time_interval_begin_hour', 'time_interval_minutes', 'travel_time']]
        gy_teample_sub = pd.merge(gy_teample_sub, test_result, on=['link_ID', 'time_interval_day', 'time_interval_begin_hour',
                                                            'time_interval_minutes'], how="left")
        gy_teample_sub.to_csv('sub_data/333.txt', index=False)

        gy_teample_sub[['link_ID', 'date_time', 'time_interval', 'travel_time']].to_csv(
            'sub_data/Fighting666666_0915_4.txt', sep='#', index=False, header=False)
        print(gy_teample_sub[['link_ID', 'date_time', 'time_interval', 'travel_time']].shape)
        print(gy_teample_sub[['link_ID', 'date_time', 'time_interval', 'travel_time']].isnull().sum())
        print("完成...")


def runLGBM(train_X, train_y, test_X, test_y,on_off):
    """
    :param train_X:  训练集数据
    :param train_y:  训练集标签 log
    :param test_X:   测试集数据
    :param test_y:   测试集标签 log
    :return:
    """
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression_l1',
        # 'metric': 'l1',
        'num_leaves': 127,
        'learning_rate': 0.01,
         #'feature_fraction': 0.9,
         #'bagging_fraction':  0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'min_data_in_leaf': 600
    }

    if on_off == "off":

        lgb_train = lgb.Dataset(train_X, train_y, silent=True)
        lgb_eval = lgb.Dataset(test_X, test_y, reference=lgb_train)
        model_ = lgb.train(params,
                           lgb_train,
                           num_boost_round=2000,
                           fobj=mape_object,
                           feval=mape_ln,
                           valid_sets=lgb_eval,
                           verbose_eval=True,
                           early_stopping_rounds=5)

        pred_test_y = model_.predict(test_X, num_iteration=model_.best_iteration)

    if on_off == "on":

        lgb_train = lgb.Dataset(train_X, train_y, silent=True)

        model_ = lgb.train(params,
                           lgb_train,
                           num_boost_round=296,
                           fobj=mape_object,
                           feval=mape_ln
                           )

        pred_test_y = model_.predict(test_X, num_iteration=model_.best_iteration)

    return pred_test_y






