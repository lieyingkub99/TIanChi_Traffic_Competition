#-----------------------------------------------------------------------------------------------------------------------
#@创建于2017-9
#@Author：lieying
#@School:USTB
#@E-mil:kwj123321@163.com
#----
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
import time

def submition_time(start_y, start_m, start_d, start_h, start_min, start_s, days, hours):
    """
    作用： 生成 从指定时间开始的提交数据，包含所有时间、所有link
    持续天数，单位是天  例： days = 1, 则 time_counts *days
    """
    time_counts = 30 * hours  # (1小时30次)
    # link_id
    all_link = pd.read_csv('pre_data/gy_contest_link_top.txt', sep=';')
    all_link = all_link['link_ID']

    # 时间
    init_time = datetime(start_y, start_m, start_d, start_h, start_min, start_s)
    link_id_list = []
    time_list1 = []
    time_list2 = []
    for i in range(len(all_link)):
        day_times = 0
        for j in range(days):             # 有多少天
            s_1 = init_time + timedelta(days=day_times)

            for k in range(time_counts):          # 有多少小时,一小时是30次
                s_2 = (s_1 + timedelta(minutes=2))      # 分钟的循环内就只加分钟，天数在分钟循环外加1就好了，加1在天数的循环内
                s_1_string = s_1.strftime('%Y-%m-%d %H:%M:%S')
                s_2_string = s_2.strftime('%Y-%m-%d %H:%M:%S')
                link_id_list.append(all_link[i])
                time_list1.append(s_1_string[:10])
                time_list2.append('[' + s_1_string + ',' + s_2_string + ')')
                s_1 = s_2
            day_times = day_times + 1

    subm = pd.DataFrame({'link_ID': link_id_list, 'date_time': time_list1,
                         'time_interval': time_list2, 'travel_time': 0},
                        columns=['link_ID', 'date_time', 'time_interval', 'travel_time'])

    return subm


def AddBaseTimeFeature(df):

    df['time_interval_begin'] = pd.to_datetime(df['time_interval'].map(lambda x: x[1:20]))

    df['time_interval_day'] = df['time_interval_begin'].map(lambda x: x.day)
    df.time_interval_day = df.time_interval_day.astype("int")
    df['time_interval_begin_hour'] = df['time_interval_begin'].map(lambda x: x.strftime('%H'))
    df.time_interval_begin_hour = df.time_interval_begin_hour.astype("int")
    df['time_interval_minutes'] = df['time_interval_begin'].map(lambda x: x.strftime('%M'))
    df.time_interval_minutes = df.time_interval_minutes.astype("int")
    df = df.drop(['time_interval_begin','travel_time'], axis=1)
    return df
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
此界面只用于生成提交的表格式
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':


    #start = time.clock()

    sub1 = submition_time(2017, 7, 1, 8, 0, 0, 31, 1)
    sub2 = submition_time(2017, 7, 1, 15, 0, 0, 31, 1)
    sub3 = submition_time(2017, 7, 1, 18, 0, 0, 31, 1)
    sub = pd.concat([sub1, sub2, sub3])
    sub = sub.sort_values(by=['link_ID', 'time_interval'])
    print(sub.shape)
    sub.to_csv('pre_data/submition_template_seg2.txt', sep=';',index=False)

    #end = time.clock()
    gy_team_sub = pd.read_csv('pre_data/submition_template_seg2.txt', sep=';', low_memory=False)

    gy_team_sub = AddBaseTimeFeature(gy_team_sub)
    gy_team_sub.to_csv('pre_data/gy_teample_sub_seg2.txt', index=False)





