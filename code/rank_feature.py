import argparse
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from pandarallel import pandarallel

from utils import Logger

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

pandarallel.initialize()

warnings.filterwarnings('ignore')

seed = 2020

# 命令行参数
parser = argparse.ArgumentParser(description='排序特征')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'排序特征，mode: {mode}')


def func_if_sum(x):
    user_id = x['user_id']
    mrch_id = x['mrch_id']

    interacted_items = user_item_dict[user_id]
    interacted_items = interacted_items[::-1]

    sim_sum = 0
    for loc, i in enumerate(interacted_items):
        try:
            sim_sum += item_sim[i][mrch_id] * (0.7**loc)
        except Exception as e:
            pass
    return sim_sum


def func_if_last(x):
    user_id = x['user_id']
    mrch_id = x['mrch_id']

    last_item = user_item_dict[user_id][-1]

    sim = 0
    try:
        sim = item_sim[last_item][mrch_id]
    except Exception as e:
        pass
    return sim


def func_binetwork_sim_last(x):
    user_id = x['user_id']
    mrch_id = x['mrch_id']

    last_item = user_item_dict[user_id][-1]

    sim = 0
    try:
        sim = binetwork_sim[last_item][mrch_id]
    except Exception as e:
        pass
    return sim


def consine_distance(vector1, vector2):
    if type(vector1) != np.ndarray or type(vector2) != np.ndarray:
        return -1
    distance = np.dot(vector1, vector2) / \
        (np.linalg.norm(vector1)*(np.linalg.norm(vector2)))
    return distance


def func_w2w_sum(x, num):
    user_id = x['user_id']
    mrch_id = x['mrch_id']

    interacted_items = user_item_dict[user_id]
    interacted_items = interacted_items[::-1][:num]

    sim_sum = 0
    for loc, i in enumerate(interacted_items):
        try:
            sim_sum += consine_distance(mrch_vec_map[mrch_id],
                                        mrch_vec_map[i])
        except Exception as e:
            pass
    return sim_sum


def func_w2w_last_sim(x):
    user_id = x['user_id']
    mrch_id = x['mrch_id']

    last_item = user_item_dict[user_id][-1]

    sim = 0
    try:
        sim = consine_distance(mrch_vec_map[mrch_id],
                               mrch_vec_map[last_item])
    except Exception as e:
        pass
    return sim


if __name__ == '__main__':
    if mode == 'valid':
        df_feature = pd.read_pickle('../user_data/data/offline/recall.pkl')
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')

    else:
        df_feature = pd.read_pickle('../user_data/data/online/recall.pkl')
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')

    # 商户特征
    log.debug(f'df_feature.shape: {df_feature.shape}')

    df_mrch = pd.read_csv('../tcdata/marchant.csv')
    
    # 历史记录相关特征
    df_click.sort_values(['user_id', 'action_time'], inplace=True)
    df_click.rename(columns={'mrch_id': 'mrch_id'}, inplace=True)
    df_click = df_click.merge(df_mrch, how='left')

    # df_click['action_time'] = df_click['action_time'] / 1000
    df_click['action_time'] = pd.to_datetime(df_click['action_time'],
                                                unit='s',
                                                errors='coerce')
    # df_click['click_datetime_hour'] = df_click['click_datetime'].dt.hour

    
    # 用户点击商户的时间差的平均值
    df_click['user_id_click_diff'] = df_click.groupby(
        ['user_id'])['action_time'].diff()
    df_temp = df_click.groupby(['user_id'
                                ])['user_id_click_diff'].mean().reset_index()
    df_temp.columns = ['user_id', 'user_id_click_diff_mean']
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')


    # 点击的商户的 brand_nm 统计值
    df_temp = df_click.groupby(['user_id'])['words_count'].agg({
        'user_clicked_mrch_words_count_mean':
        'mean',
        'user_click_last_mrch_words_count':
        lambda x: x.iloc[-1]
    }).reset_index()
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    
    # 点击的商户的 action_time 统计值
    df_temp = df_click.groupby('user_id')['action_time'].agg({
        'user_click_last_mrch_click_time':
        lambda x: x.iloc[-1],
        'user_clicked_mrch_click_time_mean':
        'mean',
    }).reset_index()
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    df_feature['user_last_click_created_at_ts_diff'] = df_feature[
        'created_at_ts'] - df_feature['user_click_last_mrch_created_time']
    df_feature['user_last_action_time_diff'] = df_feature[
        'created_at_ts'] - df_feature['user_click_last_mrch_click_time']
    df_feature['user_last_click_words_count_diff'] = df_feature[
        'words_count'] - df_feature['user_click_last_mrch_words_count']

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 计数统计
    for f in [['user_id'], ['mrch_id'], ['user_id', 'brand_nm']]:
        df_temp = df_click.groupby(f).size().reset_index()
        df_temp.columns = f + ['{}_cnt'.format('_'.join(f))]

        df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 召回相关特征
    ## itemcf 相关
    user_item_ = df_click.groupby('user_id')['mrch_id'].agg(
        list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['mrch_id']))

    if mode == 'valid':
        f = open('../user_data/sim/offline/itemcf_sim.pkl', 'rb')
        item_sim = pickle.load(f)
        f.close()
    else:
        f = open('../user_data/sim/online/itemcf_sim.pkl', 'rb')
        item_sim = pickle.load(f)
        f.close()

    # 用户历史点击物品与待预测物品相似度
    df_feature['user_clicked_mrch_itemcf_sim_sum'] = df_feature[[
        'user_id', 'mrch_id'
    ]].parallel_apply(func_if_sum, axis=1)
    df_feature['user_last_click_mrch_itemcf_sim'] = df_feature[[
        'user_id', 'mrch_id'
    ]].parallel_apply(func_if_last, axis=1)

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    ## binetwork 相关
    if mode == 'valid':
        f = open('../user_data/sim/offline/binetwork_sim.pkl', 'rb')
        binetwork_sim = pickle.load(f)
        f.close()
    else:
        f = open('../user_data/sim/online/binetwork_sim.pkl', 'rb')
        binetwork_sim = pickle.load(f)
        f.close()

    df_feature['user_last_click_mrch_binetwork_sim'] = df_feature[[
        'user_id', 'mrch_id'
    ]].parallel_apply(func_binetwork_sim_last, axis=1)

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    ## w2v 相关
    if mode == 'valid':
        f = open('../user_data/data/offline/mrch_w2v.pkl', 'rb')
        mrch_vec_map = pickle.load(f)
        f.close()
    else:
        f = open('../user_data/data/online/mrch_w2v.pkl', 'rb')
        mrch_vec_map = pickle.load(f)
        f.close()

    df_feature['user_last_click_mrch_w2v_sim'] = df_feature[[
        'user_id', 'mrch_id'
    ]].parallel_apply(func_w2w_last_sim, axis=1)
    df_feature['user_click_mrch_w2w_sim_sum_2'] = df_feature[[
        'user_id', 'mrch_id'
    ]].parallel_apply(lambda x: func_w2w_sum(x, 2), axis=1)

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 保存特征文件
    if mode == 'valid':
        df_feature.to_pickle('../user_data/data/offline/feature.pkl')

    else:
        df_feature.to_pickle('../user_data/data/online/feature.pkl')
