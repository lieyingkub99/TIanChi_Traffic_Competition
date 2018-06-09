#coding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def TimeFeature(df):

    df['time_interval_begin'] = pd.to_datetime(df['time_interval'].map(lambda x: x[1:20]))
    df = df.drop(['date_time', 'time_interval'], axis=1)
    df['time_interval_year'] = df['time_interval_begin'].map(lambda x: x.strftime('%y'))
    df.time_interval_year = df.time_interval_year.astype("int")
    df['time_interval_month'] = df['time_interval_begin'].map(lambda x: x.strftime('%m'))
    df.time_interval_month = df.time_interval_month.astype("int")
    df['time_interval_day'] = df['time_interval_begin'].map(lambda x: x.day)
    df.time_interval_day = df.time_interval_day.astype("int")
    df['time_interval_begin_hour'] = df['time_interval_begin'].map(lambda x: x.strftime('%H'))
    df.time_interval_begin_hour = df.time_interval_begin_hour.astype("int")
    df['time_interval_minutes'] = df['time_interval_begin'].map(lambda x: x.strftime('%M'))
    df.time_interval_minutes = df.time_interval_minutes.astype("int")
    # Monday=1, Sunday=7
    df['time_interval_week'] = df['time_interval_begin'].map(lambda x: x.weekday() + 1)
    df_out = df.loc[((df.time_interval_begin_hour==6) | (df.time_interval_begin_hour==7)
                   |(df.time_interval_begin_hour==8)  | (df.time_interval_begin_hour==13)
                   |(df.time_interval_begin_hour==14) | (df.time_interval_begin_hour==15)
                   |(df.time_interval_begin_hour==16) | (df.time_interval_begin_hour==17)
                   |(df.time_interval_begin_hour==18))&(df.time_interval_year==17)]

    return df_out

link_info = pd.read_csv('pre_data/gy_contest_link_info.txt',sep=';',low_memory=False)
link_info = link_info.sort_values('link_ID')
#print(link_info.info())
training_data = pd.read_csv('pre_data/quaterfinal_gy_cmp_training_traveltime.txt',sep=';',header= 0 ,low_memory=False)
training_data.columns = ['link_ID','date_time','time_interval','travel_time']
print(training_data)
#print(training_data.info())

training_data = training_data.sort_values(by=['link_ID','time_interval'])
training_data = pd.merge(training_data,link_info,on='link_ID', how='left')
print("==============原始数据行数列数==========",training_data.shape)
testing_data = pd.read_csv('pre_data/submition_template_seg2.txt',sep=';',low_memory=False)
#=======预测数据+道路基本信息合成======
testing_data = pd.merge(testing_data, link_info, on='link_ID', how='left')
print("==============7月测试数据行数列数==========",testing_data.shape)
feature_date = pd.concat([training_data,testing_data],axis=0)
feature_date = feature_date.sort_values(['link_ID','time_interval'])
print("==============原始（456）与7月测试数据行数列数==========",feature_date.shape)
feature_date.to_csv('pre_data/feature_data.txt',index=False)

feature_data = pd.read_csv('pre_data/feature_data.txt',low_memory=False)
feature_data_date = TimeFeature(feature_data)
print("==============时间分离以后，行列数==========",feature_data_date.shape)
print(feature_data_date.info())
feature_data_date.to_csv('pre_data/data_after_handle.txt',index=False)

feature_data = pd.read_csv('pre_data/data_after_handle.txt',low_memory=False)
week = pd.get_dummies(feature_data['time_interval_week'],prefix='week')
feature_data_basic_week = pd.concat([feature_data,week],axis=1)
print("==============加入week以后，行列数==========",feature_data_basic_week.shape)
feature_data_basic_week.to_csv('pre_data/feature_data_basic_week.txt',index=False)

feature_data = pd.read_csv('pre_data/feature_data_basic_week.txt',low_memory=False)
link_info_handle = pd.read_csv('data/link_info_handle.txt',low_memory=False)
feature_data = pd.merge(feature_data, link_info_handle, on='link_ID', how='left')
print("==============加入上个路口信息后行列数==========",feature_data.shape)

print(feature_data.head)

#==4月
feature_4 = feature_data.loc[(feature_data.time_interval_month == 4)
                                &((feature_data.time_interval_begin_hour ==6) | (feature_data.time_interval_begin_hour == 7)
                               | (feature_data.time_interval_begin_hour ==13) | (feature_data.time_interval_begin_hour == 14)
                               | (feature_data.time_interval_begin_hour ==16) | (feature_data.time_interval_begin_hour == 17)) ]


feature_4.to_csv('data/feature_4_seg2.txt',index=False,)
print("========feature_4的行列数为:  ",feature_4.shape)
train_4 = feature_data.loc[(feature_data.time_interval_month == 4)
                                &((feature_data.time_interval_begin_hour ==8)
                               | (feature_data.time_interval_begin_hour ==15)
                               | (feature_data.time_interval_begin_hour ==18))]

train_4.to_csv('data/train_4_seg2.txt',index=False)
print("========四月提取完成,train_4的行列数为:  ",train_4.shape)
#==五月
feature_5 = feature_data.loc[(feature_data.time_interval_month == 5)
                                &((feature_data.time_interval_begin_hour ==6) | (feature_data.time_interval_begin_hour == 7)
                               | (feature_data.time_interval_begin_hour ==13) | (feature_data.time_interval_begin_hour == 14)
                               | (feature_data.time_interval_begin_hour ==16) | (feature_data.time_interval_begin_hour == 17)) ]


feature_5.to_csv('data/feature_5_seg2.txt',index=False)
print("========feature_5的行列数为:  ",feature_5.shape)
train_5 = feature_data.loc[(feature_data.time_interval_month == 5)
                                &((feature_data.time_interval_begin_hour ==8)
                               | (feature_data.time_interval_begin_hour ==15)
                               | (feature_data.time_interval_begin_hour ==18))]

train_5.to_csv('data/train_5_seg2.txt',index=False)
print("========五月提取完成,train_5的行列数为:  ",train_5.shape)
#==6月
feature_6 = feature_data.loc[(feature_data.time_interval_month == 6)
                                &((feature_data.time_interval_begin_hour ==6) |  (feature_data.time_interval_begin_hour == 7)
                               | (feature_data.time_interval_begin_hour ==13) | (feature_data.time_interval_begin_hour == 14)
                               | (feature_data.time_interval_begin_hour ==16) | (feature_data.time_interval_begin_hour == 17))]

feature_6.to_csv('data/feature_6_seg2.txt',index=False)
print("========feature_6的行列数为:  ",feature_6.shape)
train_6 = feature_data.loc[(feature_data.time_interval_month == 6)
                              & ((feature_data.time_interval_begin_hour == 8)
                              | (feature_data.time_interval_begin_hour == 15)
                              | (feature_data.time_interval_begin_hour == 18))]

train_6.to_csv('data/train_6_seg2.txt',index=False)
print("========六月提取完成,train_6的行列数为:  ",train_6.shape)
#==7月
feature_7 = feature_data.loc[(feature_data.time_interval_month == 7)
                              &((feature_data.time_interval_begin_hour ==6) |  (feature_data.time_interval_begin_hour == 7)
                             | (feature_data.time_interval_begin_hour ==13) | (feature_data.time_interval_begin_hour == 14)
                             | (feature_data.time_interval_begin_hour ==16 )| (feature_data.time_interval_begin_hour == 17) ) ]

feature_7.to_csv('data/feature_7_seg2.txt',index=False)
print("========feature_7的行列数为:  ",feature_7.shape)
train_7 = feature_data.loc[(feature_data.time_interval_month == 7)
                                &((feature_data.time_interval_begin_hour ==8)
                               | (feature_data.time_interval_begin_hour ==15)
                               | (feature_data.time_interval_begin_hour ==18))]

train_7.to_csv('data/train_7_seg2.txt',index=False)
print("========七月提取完成,feature_7train_7的行列数为:  ",train_7.shape)


