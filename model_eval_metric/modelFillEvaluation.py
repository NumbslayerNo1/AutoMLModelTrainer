import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn import metrics
import plotnine
from plotnine import *
import pandarallel



'''
1. 分布
'''
'''
1.1 分bin分布
'''
def _get_bin_dist(df: pd.core.frame.DataFrame, ori_score: str, fill_score_list: list, bin_num: int = 10) -> pd.core.frame.DataFrame:
    
    '''
    将base模型等频分Bin的阈值映射到对比模型分上, 查看各阈值下的分布变化
    
    Parameters
    ----------
    
    df：包含样本的各列模型分
    ori_score：base模型分
    fill_score_list：需要对比的模型columns_list
    bin_num：等频分箱数
    return：按base模型等频分箱的样本量，各模型的分箱占比，累积分箱占比，累积分箱占比diff
    '''
    
    df[f'bins_{ori_score}'], break_list  = pd.qcut(df[ori_score], bin_num, duplicates = 'drop', retbins = True)
    break_list[0] = -np.inf
    break_list[-1] = np.inf
    
    df[f'bins_{ori_score}'] = pd.cut(df[ori_score], break_list)
    for k in fill_score_list:
        df[f'bins_{k}'] = pd.cut(df[k], break_list)
        
    rs_df = df.groupby(f'bins_{ori_score}').agg(
        {
            'trace_id': 'count'
        }
    ).reset_index()
    rs_df.columns = ['bins', f'{ori_score}_cnt']
    rs_df[f'{ori_score}_ratio'] = rs_df[f'{ori_score}_cnt'] / rs_df[f'{ori_score}_cnt'].sum()
        
    for k in fill_score_list:
        agg_df_k = df.groupby(f'bins_{k}').agg(
            {
                'trace_id': 'count'
            }
        ).reset_index()
        agg_df_k.columns = ['bins', f'{k}_cnt']
        agg_df_k[f'{k}_ratio'] = agg_df_k[f'{k}_cnt'] / agg_df_k[f'{k}_cnt'].sum()
        
        rs_df = pd.merge(
            rs_df,
            agg_df_k,
            how = 'outer',
            on = 'bins'
        ).reset_index(drop = True)
        
        rs_df[f'{k}_ratio_diff'] = agg_df_k[f'{k}_ratio'] - rs_df[f'{ori_score}_ratio']
        rs_df[f'{k}_ratio_diff_cum'] = rs_df[f'{k}_ratio_diff'].cumsum()
        
    return rs_df

def get_groups_bin_dist(df: pd.core.frame.DataFrame, ori_score: str, fill_score_list: list, group: str, bin_num: int = 10) -> pd.core.frame.DataFrame:
    
    '''
    每组的模型分分布变化情况
    
    Parameters
    ----------
    
    df：包含样本的各列模型分
    ori_score：base模型分
    fill_score_list：需要对比的模型columns_list
    group：组别
    bin_num：等频分箱数
    return：每组内，按base模型等频分箱的样本量，各模型的分箱占比，累积分箱占比，累积分箱占比diff
    '''
    
    pandarallel.pandarallel.initialize(nb_workers = 10, progress_bar = True)
    rs_df = df.groupby(group).parallel_apply(lambda x: _get_bin_dist(x, ori_score, fill_score_list, bin_num)).reset_index()
    
    display(
        rs_df[
            [
                group,
                'bins',
                f'{ori_score}_cnt',
                f'{ori_score}_ratio'
            ] +\
            [f'{k}_ratio' for k in fill_score_list] +\
            [f'{k}_ratio_diff' for k in fill_score_list] +\
            [f'{k}_ratio_diff_cum' for k in fill_score_list]
        ]
    )
    
    return rs_df



'''
1.2 累积分布图
'''
def _get_cum_dist(df: pd.core.frame.DataFrame, score: str) -> pd.core.frame.DataFrame:
    
    '''
    计算一个模型，各分值下的累积占比
    
    Parameters
    ----------
    
    df：包含样本的模型分
    score：模型列名
    return：
    '''
    
    sorted_list = sorted(df[score])
    
    # 1002个点
    avg_cnt = len(sorted_list) // 1000
    
    rs_df = defaultdict(list)
    for i in range(len(sorted_list)):
        
        if i == 0:
            
            rs_df[f'score'].append(sorted_list[i])
            rs_df['cum_proportion'].append((i + 1) / len(sorted_list))
            
        elif i == len(sorted_list) - 1:
            
            rs_df['score'].append(sorted_list[i])
            rs_df['cum_proportion'].append((i + 1) / len(sorted_list))
            
        else:
            
            if (i + 1) // avg_cnt < (i + 2) // avg_cnt:
                
                rs_df['score'].append(sorted_list[i])
                rs_df['cum_proportion'].append((i + 1) / len(sorted_list))
                
    rs_df = pd.DataFrame(rs_df)
    rs_df['model'] = score
        
    return rs_df

def get_scores_cum_dist(df: pd.core.frame.DataFrame, score_list: list) -> pd.core.frame.DataFrame:
    
    '''
    计算多个模型，各分值下的累积占比
    
    Parameters
    ----------
    
    df：包含样本的各模型分
    score_list：需要计算的模型columns_list
    return：
    '''
    
    rs_df = []
    for score in score_list:
        rs_df.append(_get_cum_dist(df, score))
        
    rs_df = pd.concat(rs_df).reset_index(drop = True)
    
    return rs_df

def get_groups_cum_dist(df: pd.core.frame.DataFrame, score_list: list, group: str) -> pd.core.frame.DataFrame:
    
    '''
    计算每组内，多个模型，各分值下的累积占比，并画出累积分布曲线
    
    Parameters
    ----------
    
    df：包含样本的各模型分
    score_list：需要计算的模型columns_list
    group：组别
    return：画累积分布曲线的DataFrame
    '''
    
    pandarallel.pandarallel.initialize(nb_workers = 10, progress_bar = True)
    rs_df = df.groupby(group).parallel_apply(lambda x: get_scores_cum_dist(x, score_list)).reset_index()
    
    plotnine.options.figure_size = (16, 8)
    print(
    ggplot(data = rs_df, mapping = aes(x = 'score', y = 'cum_proportion', color = 'model'))
    + geom_line()
    + facet_wrap('trace_month')
)
    
    return rs_df



'''
2. 风险
'''
'''
2.1 AUC
''' 
def _get_auc(df: pd.core.frame.DataFrame, ori_score: str, fill_score_list: list, label: str) -> pd.core.frame.DataFrame:
    
    '''
    计算base模型和对比模型的auc，并计算auc差异
    
    Parameters
    ----------
    
    df：包含样本的各列模型分和label
    ori_score：base模型分
    fill_score_list：需要对比的模型columns_list
    label：计算auc的指标，(0, 1)
    return：base模型和对比模型的auc，以及auc_diff
    '''
    
    rs_df = {}
    rs_df['cnt'] = [df.shape[0]]
    
    rs_df[f'{ori_score}_auc_{label}'] = [metrics.roc_auc_score(df[label], df[ori_score])]
    for k in fill_score_list:
        rs_df[f'{k}_auc_{label}'] = [metrics.roc_auc_score(df[label], df[k])]
        
    rs_df = pd.DataFrame(rs_df)
    
    for k in fill_score_list:
        rs_df[f'{k}_auc_diff_{label}'] = rs_df[f'{k}_auc_{label}'] - rs_df[f'{ori_score}_auc_{label}']
        
    return rs_df

def get_groups_auc(df: pd.core.frame.DataFrame, ori_score: str, fill_score_list: list, label: str, group: str) -> pd.core.frame.DataFrame:
    
    '''
    每组的模型分auc变化情况
    
    Parameters
    ----------
    
    df：包含样本的各列模型分和label
    ori_score：base模型分
    fill_score_list：需要对比的模型columns_list
    label：计算auc的指标，(0, 1)
    group：组别
    return：每组内，base模型和对比模型的auc，以及auc_diff
    '''
    
    pandarallel.pandarallel.initialize(nb_workers = 10, progress_bar = True)
    rs_df = df.groupby(group).parallel_apply(lambda x: _get_auc(x, ori_score, fill_score_list, label)).reset_index()
    
    display(
        rs_df[
            [
                group,
                'cnt',
                f'{ori_score}_auc_{label}'
            ] +\
            [f'{k}_auc_{label}' for k in fill_score_list] +\
            [f'{k}_auc_diff_{label}' for k in fill_score_list]
        ]
    )
    
    return rs_df



'''
2.2 分bin风险
''' 
def _get_bin_risk(df: pd.core.frame.DataFrame, ori_score: str, fill_score_list: list, label: str, bin_num: int = 10) -> pd.core.frame.DataFrame:
    
    '''
    base模型和对比模型分别等频分箱，计算相同通过率的情况下风险差异
    
    Parameters
    ----------
    
    df：包含样本的各列模型分和label
    ori_score：base模型分
    fill_score_list：需要对比的模型columns_list
    label：计算风险指标，(0, 1)
    bin_num：等频分箱数
    return：base模型和对比模型的同通过率下（单bin、累计bin）的风险，风险diff
    '''
    
    df[f'qbins_{ori_score}'] = pd.qcut(df[ori_score], bin_num, labels = False) + 1
    
    rs_df = df.groupby(f'qbins_{ori_score}').agg(
        {
            'trace_id': 'count',
            label: 'mean'
        }
    ).reset_index()
    rs_df.columns = ['qbins', 'bin_cnt', f'{ori_score}_{label}']
    
    rs_df = pd.concat(
        [
            rs_df,
            pd.DataFrame(
                [
                    {
                        'qbins': 'total', 
                        'bin_cnt': rs_df['bin_cnt'].sum(),
                        f'{ori_score}_{label}': df[label].mean()
                    }
                ]
            )
        ]
    ).reset_index(drop = True)
        
    for k in fill_score_list:
        df[f'qbins_{k}'] = pd.qcut(df[k], bin_num, labels = False) + 1
        agg_df_k = df.groupby(f'qbins_{k}').agg(
            {
                label: 'mean'
            }
        ).reset_index()
        agg_df_k.columns = ['qbins', f'{k}_{label}']
        
        agg_df_k = pd.concat(
            [
                agg_df_k,
                pd.DataFrame(
                    [
                        {
                            'qbins': 'total',
                            f'{k}_{label}': df[label].mean(),
                        }
                    ]
                )
            ]
        ).reset_index(drop = True)
        
        rs_df = pd.merge(
            rs_df,
            agg_df_k,
            how = 'outer',
            on = 'qbins'
        ).reset_index(drop = True)
        
        rs_df[f'{k}_{label}_diff'] = agg_df_k[f'{k}_{label}'] - rs_df[f'{ori_score}_{label}']
        rs_df[f'{k}_{label}_diff_cum'] = rs_df[f'{k}_{label}_diff'].cumsum()
        
    return rs_df

def get_groups_bin_risk(df: pd.core.frame.DataFrame, ori_score: str, fill_score_list: list, group: str, label: str, bin_num: int = 10) -> pd.core.frame.DataFrame:
    
    '''
    每组的，base模型和对比模型分别等频分箱，计算相同通过率的情况下风险差异
    
    Parameters
    ----------
    
    df：包含样本的各列模型分和label
    ori_score：base模型分
    fill_score_list：需要对比的模型columns_list
    group：组别
    label：计算风险指标，(0, 1)
    bin_num：等频分箱数
    return：每组内，base模型和对比模型的同通过率下（单bin、累计bin）的风险，风险diff
    '''
    
    pandarallel.pandarallel.initialize(nb_workers = 10, progress_bar = True)
    rs_df = df.groupby(group).parallel_apply(lambda x: _get_bin_risk(x, ori_score, fill_score_list, label, bin_num)).reset_index()
    
    display(
        rs_df[
            [
                group,
                'qbins',
                f'{ori_score}_{label}',
            ] +\
            [f'{k}_{label}' for k in fill_score_list] +\
            [f'{k}_{label}_diff' for k in fill_score_list] +\
            [f'{k}_{label}_diff_cum' for k in fill_score_list]
        ]
    )
    
    return rs_df


