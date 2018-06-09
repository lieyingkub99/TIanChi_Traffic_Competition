#-----------------------------------------------------------------------------------------------------------------------
#@创建于2017-9
#@Author：lieying
#@School:USTB
#@E-mil:kwj123321@163.com
#----
import pandas as pd
import numpy as np
import math
def offline_confuse(i):
    if  i=="off":
        off_1 = pd.read_csv('sub_data/model_915_offline.txt',low_memory=False)
        off_1 = off_1.sort_values(by=['link_ID', 'time_interval_day', 'time_interval_begin_hour', 'time_interval_minutes'])
        off_2 = pd.read_csv('test/june_lgb_1_all_2016_2.csv',header=0, sep=';',low_memory=False)
        off_2 = off_2.rename(columns={'travel_time': 'travel_time_1'})
        off_2 = off_2.sort_values(by=['link_ID', 'time_interval'])

        off_2["travel_time_fuse"] = ( off_1["travel_time_915"]*2 + off_2["travel_time_1"] )/3
        off_2['score'] = abs(off_2['travel_time_fuse'] - off_1['travel_time']) / off_1['travel_time']
        score = sum(off_2['score'])/off_1.shape[0]
        print("线下测试分数是： ", score)
    if  i=="on":
        #-融合
        #print("confuse  is  not beginning  please check .........")
        on_1 = pd.read_csv('sub_data/Fighting666666_0915_1.txt', header=None, sep='#', low_memory=False)
        on_1.columns = ['link_ID', 'date_time', 'time_interval', 'travel_time']
        on_1 = on_1.rename(columns={'travel_time': 'travel_time_1'})

        on_2 = pd.read_csv('sub_data/Fighting666666_0915_2.txt',header=None, sep='#',  low_memory=False)
        on_2.columns = ['link_ID', 'date_time', 'time_interval', 'travel_time']
        on_2 = on_2.rename(columns={'travel_time': 'travel_time_2'})

        on_3 = pd.read_csv('sub_data/Fighting666666_0915_3.txt', header=None, sep='#', low_memory=False)
        on_3.columns = ['link_ID', 'date_time', 'time_interval', 'travel_time']
        on_3 = on_2.rename(columns={'travel_time': 'travel_time_3'})

        on_4 = pd.read_csv('sub_data/Fighting666666_0915_4.txt', header=None, sep='#', low_memory=False)
        on_4.columns = ['link_ID', 'date_time', 'time_interval', 'travel_time']
        on_4 = on_2.rename(columns={'travel_time': 'travel_time_4'})

        online = pd.merge(on_1, on_2,on_3,on_4,on=['link_ID','date_time','time_interval'],how="left")

        #-存档
        online["travel_time"] = (online["travel_time_1"]*2 + online["travel_time_2"]*2+online["travel_time_3"]*3 + online["travel_time_4"]*3) / 10
        online[['link_ID', 'date_time', 'time_interval', 'travel_time']].to_csv('sub_data/Fighting666666_sub_end.txt', sep='#', index=False, header=False)

        #-与最好比较MAPE
        # =读预测和以前较好的数据
        test_1 = pd.read_table('sub_data/Fighting666666_0911.txt', header=None, sep='#', low_memory=False)
        test_1.columns = ['link_ID', 'date_time', 'time_interval', 'travel_time']
        print(test_1)
        print(test_1[['link_ID', 'date_time', 'time_interval', 'travel_time']].isnull().sum())
        test_2 = pd.read_table('sub_data/Fighting666666_sub_end.txt', header=None, sep='#', low_memory=False)
        test_2.columns = ['link_ID', 'date_time', 'time_interval', 'travel_time']
        print(test_2)
        # =与之前的，测MAPE
        test_1 = test_1.sort_values(by=['link_ID', 'time_interval'])
        test_2 = test_2.sort_values(by=['link_ID', 'time_interval'])
        result = np.sum(np.abs(test_2.travel_time - test_1.travel_time))
        print("与上次最好相比MAPE为：  ", result)

        # =============================提交文件小于10M，必须分开===============================
        '''
        spilt_sub = pd.read_table('sub_data/Fighting666666_sub_end.txt',sep='#',header=None,low_memory=False)
        spilt_sub.columns = ['link_ID', 'date_time', 'time_interval', 'travel_time']
        spilt_sub_1 = spilt_sub.loc[spilt_sub.date_time<="2017-07-08"]
        spilt_sub_1.to_csv('sub_data/spilt_sub_1_0914.txt',sep='#',index=False)
        spilt_sub_2 = spilt_sub.loc[(spilt_sub.date_time>"2017-07-08")&(spilt_sub.date_time<="2017-07-15")]
        spilt_sub_2.to_csv('sub_data/spilt_sub_2_0914.txt',sep='#',index=False)
        spilt_sub_3 = spilt_sub.loc[(spilt_sub.date_time>"2017-07-15")&(spilt_sub.date_time<="2017-07-23")]
        spilt_sub_3.to_csv('sub_data/spilt_sub_3_0914.txt', sep='#', index=False)
        spilt_sub_4 = spilt_sub.loc[(spilt_sub.date_time>"2017-07-23")&(spilt_sub.date_time<="2017-07-31")]
        spilt_sub_4.to_csv('sub_data/spilt_sub_4_0914.txt', sep='#', index=False)
        print("完成！")
        '''



