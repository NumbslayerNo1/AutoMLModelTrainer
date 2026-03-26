import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn import metrics
import plotnine
from plotnine import *
import pandarallel
import lightgbm as lgb
import copy
import json

'''
1. PSI计算
'''
def get_psi(x1: list, x2: list) -> float:
    '''
    计算psi
    
    Parameters
    ----------
    
    x1：预期list
    x2：实际list
    return：两者psi
    '''
    
    bins = 10
        
    # 按预期数据计算
    x1_null_cnt = sum([1 for x in x1 if pd.isnull(x)])
    x1_notnull_list = [x for x in x1 if pd.notnull(x)]
    
    if len(x1_notnull_list) > 0:
        _, cuts = pd.qcut(x = x1_notnull_list, q = bins, retbins = True, duplicates = 'drop')
        cuts = list(cuts)

        cuts[0] =  -float("inf")
        cuts[-1] = float("inf")

        expect_cuts = np.histogram(x1_notnull_list, bins = cuts)
        expect_df = pd.DataFrame([x1_null_cnt] + expect_cuts[0].tolist(), columns = ['expect'])
        
    else:
        
        raise 'expect all null'
        
    # 按实际数据计算
    x2_null_cnt = sum([1 for x in x2 if pd.isnull(x)])
    x2_notnull_list = [x for x in x2 if pd.notnull(x)]
    
    actual_cuts = np.histogram(x2_notnull_list, bins = cuts)
    actual_df = pd.DataFrame([x2_null_cnt] + actual_cuts[0].tolist(), columns = ['actual'])
    
    # 计算psi
    psi_df = pd.merge(
        expect_df,
        actual_df,
        right_index = True,
        left_index = True
    )
    
    psi_df['expect_rate'] = (psi_df['expect'] + 1) / (psi_df['expect'].sum() + bins) #计算占比，分子加1，防止计算PSI时分子分母为0
    psi_df['actual_rate'] = (psi_df['actual'] + 1) / (psi_df['actual'].sum() + bins)
    
    psi_df['psi'] = (psi_df['actual_rate'] - psi_df['expect_rate']) * np.log(
        psi_df['actual_rate'] / psi_df['expect_rate'])
    
    psi = psi_df['psi'].sum()

    return psi

'''
2. 计算降低psi的topN特征
'''
def get_psi_topn_col(
    model: lgb.basic.Booster,
    fea_list: list,
    trace_data_1: pd.core.frame.DataFrame,
    trace_data_2: pd.core.frame.DataFrame,
    result_path: str,
    max_rm_cnt: int = 20,
    min_psi: float = 0.0045
):
    '''
    计算降低psi的topN特征
    
    Parameters
    ----------
    
    model：lgb模型
    fea_list：入模特征list
    trace_data_1：预期分布的样本，包含fea_list所有特征
    trace_data_2：实际分布的样本，包含fea_list所有特征
    result_path：结果存储地址
    max_rm_cnt：设置停止条件，最多剔除的特征数topN
    min_psi：设置停止条件，psi下降到的最小值
    return：topN特征list 和 json中间结果
    '''
    
    def _get_cols_psi(tar_cols):
        x1 = trace_contir_1[tar_cols + ['base_line']].sum(axis = 1).values
        x1 = 1/(1 + np.exp([i * -1 for i in x1]))

        x2 = trace_contir_2[tar_cols + ['base_line']].sum(axis = 1).values
        x2 = 1/(1 + np.exp([i * -1 for i in x2]))

        cols_psi = get_psi(x1, x2)

        return cols_psi
    
    # 计算样本每个特征的shap值
    trace_contir_1 = pd.DataFrame(
        model.predict(trace_data_1[fea_list], pred_contrib = True)
    )
    trace_contir_1.columns = fea_list + ['base_line']

    trace_contir_2 = pd.DataFrame(
        model.predict(trace_data_2[fea_list], pred_contrib = True)
    )
    trace_contir_2.columns = fea_list + ['base_line']
    
    
    # 计算base_psi
    x1 = list(model.predict(trace_data_1[fea_list]))
    x2 = list(model.predict(trace_data_2[fea_list]))
    base_psi = get_psi(x1, x2)
    
    
    # 计算降低psi的topN特征
    pandarallel.pandarallel.initialize()
    
    cols = copy.copy(fea_list)

    # 初始化
    n = 0
    rm_col_list = []
    rs_dict = {
        'n': [n],
        'col': [None],
        'rm_col_list': [copy.copy(rm_col_list)],
        'psi': [base_psi]
    }

    t_psi = base_psi 
    while len(rm_col_list) < max_rm_cnt and t_psi > min_psi:
        n += 1

        t_col_df = pd.DataFrame(
            {
                'feature': cols
            }
        )

        t_col_df['psi'] = t_col_df['feature'].parallel_apply(lambda x: _get_cols_psi([i for i in cols if i != x]))

        t_psi = t_col_df['psi'].min()
        t_col = t_col_df[t_col_df['psi'] == t_psi]['feature'].tolist()[0]
        rm_col_list.append(t_col)

        rs_dict['n'].append(n)
        rs_dict['col'].append(t_col)
        rs_dict['rm_col_list'].append(copy.copy(rm_col_list))
        rs_dict['psi'].append(t_psi)

        with open(f"{result_path}.json", "w") as f:
            json.dump(rs_dict, f)

        print(f'{n} {t_col} {t_psi}')

        cols.remove(t_col)
        
    return rm_col_list, rs_dict
    
    