#-----------------------------------------------------------------------------------------------------------------------
#@创建于2017-9
#@Author：lieying
#@School:USTB
#@E-mil:kwj123321@163.com
#----
import pandas as pd
import numpy as np
# import main_kdd


def in_link_count(in_links):
    if in_links == 'nan':
        return 0
    else:
        return len(in_links.split('#'))

def out_link_count(out_links):
    if out_links == 'nan':
        return 0
    else:
        return len(out_links.split('#'))

def topu_1(link_topu):
    if link_topu != 'no':
        topu = link_topu.split('#')
        if len(topu)>=1:
            return topu[0]

def topu_2(link_topu):
    if link_topu != 'no':
        topu = link_topu.split('#')
        if len(topu)>=2:
            return topu[1]

def topu_3(link_topu):
    if link_topu != 'no':
        topu = link_topu.split('#')
        if len(topu)>=3:
            return topu[2]

def topu_4(link_topu):
    if link_topu != 'no':
        topu = link_topu.split('#')
        if len(topu)>=4:
            return topu[3]

def link_top_process(file='pre_data/gy_contest_link_top.txt'):
    link_top_data = pd.read_csv(file, sep=';',dtype={'link_ID':np.str})
    link_top_data = link_top_data.rename(columns={'link_ID':'link_id'})
    # link_top_data = link_top_data.astype(np.str)
    link_top_data = link_top_data.fillna('no')
    link_top_data['i_num'] = link_top_data['in_links'].apply(in_link_count)
    link_top_data['o_num'] = link_top_data['out_links'].apply(out_link_count)
    # link_top_data['i/o'] = 1.0*link_top_data['i_num']*link_top_data['o_num']
    link_top_data = link_top_data.sort_values(by='link_id')
    link_top_data = link_top_data.reset_index(0, drop=True)
    link_top_data['in_first'] = link_top_data['in_links'].apply(topu_1)
    link_top_data['in_second'] = link_top_data['in_links'].apply(topu_2)
    link_top_data['in_third'] = link_top_data['in_links'].apply(topu_3)
    link_top_data['in_forth'] = link_top_data['in_links'].apply(topu_4)
    link_top_data['out_first'] = link_top_data['out_links'].apply(topu_1)
    link_top_data['out_second'] = link_top_data['out_links'].apply(topu_2)
    link_top_data['out_third'] = link_top_data['out_links'].apply(topu_3)
    link_top_data['out_forth'] = link_top_data['out_links'].apply(topu_4)
    link_top_data = link_top_data.drop(['in_links', 'out_links'], axis=1)
    return link_top_data


def main():
    print('-----------running--------------')

if __name__ == '__main__':
    main()
    a = link_top_process()
    a.to_csv('data/link_top_process.txt', index=False)
    a = pd.read_csv('data/link_top_process.txt', sep=',',\
    dtype={'link_ID':np.str,'in_first':np.str,'in_second':np.str,'in_forth':np.str,'in_third':np.str,'out_first':np.str,'out_second':np.str,'out_forth':np.str,'out_third':np.str})
    print(a.info())
    link_info = pd.read_csv('pre_data/gy_contest_link_info.txt', sep=';')
    link_process = a.merge(link_info, left_on='in_first',  right_on='link_ID', how='left')
    link_process = link_process.rename(columns={'length': 'length_in1','width': 'width_in1','link_class': 'link_class_in1'})
    link_process = link_process.drop(['link_ID'], axis=1)
    print(link_process)

    link_process = link_process.merge(link_info, left_on='in_second', right_on='link_ID', how='left')
    link_process = link_process.rename(
        columns={'length': 'length_in2', 'width': 'width_in2', 'link_class': 'link_class_in2'})
    link_process = link_process.drop(['link_ID'], axis=1)

    link_process = link_process.merge(link_info, left_on='in_third',  right_on='link_ID', how='left')
    link_process = link_process.rename(
        columns={'length': 'length_in3', 'width': 'width_in3', 'link_class': 'link_class_in3'})
    link_process = link_process.drop(['link_ID'], axis=1)
    link_process = link_process.merge(link_info, left_on='in_forth',  right_on='link_ID', how='left')
    link_process = link_process.rename(
        columns={'length': 'length_in4', 'width': 'width_in4', 'link_class': 'link_class_in4'})
    link_process = link_process.drop(['link_ID'], axis=1)
    link_process = link_process.merge(link_info, left_on='out_first', right_on='link_ID', how='left')
    link_process = link_process.rename(
        columns={'length': 'length_out1', 'width': 'width_out1', 'link_class': 'link_class_out1'})
    link_process = link_process.drop(['link_ID'], axis=1)
    link_process = link_process.merge(link_info, left_on='out_second',right_on='link_ID', how='left')
    link_process = link_process.rename(
        columns={'length': 'length_out2', 'width': 'width_out2', 'link_class': 'link_class_out2'})
    link_process = link_process.drop(['link_ID'], axis=1)
    link_process = link_process.merge(link_info, left_on='out_third', right_on='link_ID', how='left')
    link_process = link_process.rename(
        columns={'length': 'length_out3', 'width': 'width_out3', 'link_class': 'link_class_out3'})
    link_process = link_process.drop(['link_ID'], axis=1)
    link_process = link_process.merge(link_info, left_on='out_forth', right_on='link_ID', how='left')
    link_process = link_process.rename(
        columns={'length': 'length_out4', 'width': 'width_out4', 'link_class': 'link_class_out4'})
    link_process = link_process.drop(['link_ID'], axis=1)

    link_process = link_process.rename(columns={'link_id': 'link_ID'})
    link_process.to_csv('data/link_info_handle.txt', index=False)