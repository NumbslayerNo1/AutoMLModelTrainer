# -*- coding: utf8  -*-

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn import metrics
import plotnine
from plotnine import *
from itertools import chain

import shap
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns



'''
1. 三人群区分
'''
def get_threeGroup_tag(test_data: pd.core.frame.DataFrame, train_data: pd.core.frame.DataFrame, label: str, suffix: str = ''):
    
    '''
    根据训练集判定测试集中的三个群体：TrainBad,UOOT,GOOT,BOOT(UOOT + GOOT)
    
    Parameters
    ----------
    
    test_data：测试集，需要被打tag的样本集
    train_data：模型训练使用的训练集，包括label
    label：模型训练使用的label
    suffix：给TrainBad,UOOT,GOOT,BOOT加上后缀，便于多个模型的人群的区分
    return：test_data，里面包括TrainBad,UOOT,GOOT,BOOT的tag
    '''
    
    if suffix != '':
        suffix = '_' + suffix
    
    train_account_list = list(train_data['loan_account_id'].unique())
    train_bad_account_list = list(train_data[train_data[label] == 1]['loan_account_id'].unique())
    train_good_account_list = list(set(train_account_list) - set(train_bad_account_list))
    
    
    # uoot tag
    train_account_df = pd.DataFrame(
        {
            'loan_account_id': train_account_list
        }
    )
    train_account_df[f'uoot{suffix}'] = 0
    
    test_data = pd.merge(
        test_data,
        train_account_df,
        how = 'left',
        on = 'loan_account_id'
    ).reset_index(drop = True)
    
    l = test_data[f'uoot{suffix}'].tolist()
    test_data[f'uoot{suffix}'] = [1 if pd.isnull(x) else x for x in l]
    
    
    # goot tag
    train_good_account_df = pd.DataFrame(
        {
            'loan_account_id': train_good_account_list
        }
    )
    train_good_account_df[f'goot{suffix}'] = 1
    
    test_data = pd.merge(
        test_data,
        train_good_account_df,
        how = 'left',
        on = 'loan_account_id'
    ).reset_index(drop = True)
    
    l = test_data[f'goot{suffix}'].tolist()
    test_data[f'goot{suffix}'] = [0 if pd.isnull(x) else x for x in l]
    
    
    # boot tag
    l = test_data[[f'uoot{suffix}', f'goot{suffix}']].to_dict('records')
    test_data[f'boot{suffix}'] = [1 if (x[f'uoot{suffix}'] + x[f'goot{suffix}']) > 0 else 0 for x in l]
    
    # TrainBad tag
    train_bad_account_df = pd.DataFrame(
        {
            'loan_account_id': train_bad_account_list
        }
    )
    train_bad_account_df[f'TrainBad{suffix}'] = 1
    
    test_data = pd.merge(
        test_data,
        train_bad_account_df,
        how = 'left',
        on = 'loan_account_id'
    ).reset_index(drop = True)
    
    l = test_data[f'TrainBad{suffix}'].tolist()
    test_data[f'TrainBad{suffix}'] = [0 if pd.isnull(x) else x for x in l]
    
    return test_data



'''
2. 分组计算auc
'''
def get_auc_groups(test_data: pd.core.frame.DataFrame, model_list: list, label: str, group: str):
    
    '''
    计算指定group和模型分的auc
    
    Parameters
    ----------
    
    test_data：测试集，包含模型分和label
    model_list：需要计算auc的模型list
    label：计算auc使用的label
    return：
    '''
    
    group_list = list(test_data[group].unique())
    group_list.sort()
    
    # 保证对比的样本所有模型分都有值
    eval_str = 'test_data[' + '&'.join([f"(test_data['{col}'].notnull())" for col in model_list]) + ']'
    test_data_clear = eval(eval_str)
    
    if test_data.shape[0] != test_data_clear.shape[0]:
        print('!!!注意!!! 此DataFrame存在部分样本的模型分为nan, 以下仅对所有模型分有值的样本进行结果输出')
    
    rs_df = defaultdict(list)
    
    # 分组计算
    for grp in group_list:
        df = test_data_clear[
            test_data_clear[group] == grp
        ].reset_index(drop = True)
        
        rs_df[group].append(grp)
        rs_df['count'].append(df.shape[0])
        rs_df[label].append(df[label].mean())
        
        if df[label].nunique() > 1:
            for col in model_list:
                auc = metrics.roc_auc_score(df[label], df[col])
                rs_df[col].append(auc)
                
        else:
            for col in model_list:
                rs_df[col].append(np.nan)
            
    # total计算
    rs_df[group].append('total')
    rs_df['count'].append(test_data_clear.shape[0])
    rs_df[label].append(test_data_clear[label].mean())
    for col in model_list:
        auc = metrics.roc_auc_score(test_data_clear[label], test_data_clear[col])
        rs_df[col].append(auc)
    
        
    rs_df = pd.DataFrame(rs_df)
    display(rs_df)
    
    return rs_df



'''
3. 坏账曲线
'''
def get_model_plot(df: pd.core.frame.DataFrame, 
                   unpaid: str, # 分子
                   principal: str, # 分母
                   title: str,
                   month: str, 
                   score_list: list, # score_list
                   model_name_list: list, # model_name_list
                   label_list: list, # label_list
                   bin_num: int = 20,
                   p: float = 0,
                   plot_type: str = 'all'
                  ):
    '''
    计算样本的坏账曲线
    
    Parameters
    ----------
    
    df：测试集，包含模型分、label和刻画曲线的指标
    unpaid：排序后累加的分子
    principal：排序后累加的分母
    title：设置曲线的标题
    month：样本涉及的月份
    score_list：需要对比的模型list
    model_list：对score_list创建别名，便于legend展示
    label_list：计算auc使用的label_list
    p：累积曲线从多少分位点开始刻画
    plot_type：设置图片输出类型，'all'(即 bin + cum), 'bin', 'cum'
    return：
    '''
    
    if plot_type not in ['all', 'bin', 'cum']:
        print("请输入正确的plot_type， ['all', 'bin', 'cum']")
        pass
    
    result = df.copy()
    result.reset_index(drop = True, inplace = True)
    
    # 计算AUC
    for i in range(len(score_list)):
        
        auc_dict = {}
        for label in label_list:
            auc_dict[label] = metrics.roc_auc_score(result[label], result[score_list[i]])
        
        model_name_list[i] = f'{model_name_list[i]} | ' + ' | '.join([f'{label} {round(auc_dict[label], 4)}' for label in auc_dict])
    
    if plot_type in ['all', 'bin']:
        bin_result_concat = []
        breaks_list = []
        for i in range(len(score_list)):
            result.sort_values(by = score_list[i], axis = 0, ascending = True, inplace = True)
            result[f'{score_list[i]}_bin'] = pd.qcut(result[score_list[i]], bin_num, labels = False) + 1

            bins = []
            vintage = []
            for b_i in range(1, bin_num + 1):
                bins.append(b_i)
                v = result[result[f'{score_list[i]}_bin'] == b_i][unpaid].sum() / result[result[f'{score_list[i]}_bin'] == b_i][principal].sum()
                vintage.append(v)

            df_i = pd.DataFrame({'bin': bins, 'vintage' : vintage})
            df_i['model'] = str(i + 1)
            bin_result_concat.append(df_i)
            breaks_list.append(str(i + 1))

        bin_result_concat = pd.concat(bin_result_concat)

        #plot result
        plotnine.options.figure_size = (16, 8)

        ##vintage
        print(ggplot(data = bin_result_concat, mapping = aes(x = 'bin', y = 'vintage', color = 'model'))
              + geom_line() #, alpha = 0.5
              + geom_point()
              + geom_text(aes(label = round(bin_result_concat['vintage'], 4)), position = position_dodge(width = 1), size = 8)
              + labs(x = 'bin', y = 'Vintage', title = f'{month}: {title}')
              + scale_color_hue(name = "Model Type", breaks = breaks_list, 
                               labels = model_name_list)
             )
    
    if plot_type in ['all', 'cum']:
        ##def draw_bad_debt_rate
        def getkey(element):
            return element[0]

        def draw_bad_debt_rate(pred_result):
            pred_result.sort(key = getkey)
            x = []
            y = []
            length = len(pred_result)
            principal_total = 0.
            bad_debt_total = 0.
            for i in range(length):
                x.append((i + 1.0)/length)
                principal_total = principal_total + pred_result[i][1]
                bad_debt_total = bad_debt_total + pred_result[i][2]
                y.append(bad_debt_total / principal_total)
            return x, y


        ##cumulative
        result_vintage = []
        for i in range(len(score_list)):
            model_x, model_y = draw_bad_debt_rate(list(zip(result[score_list[i]], result[principal], result[unpaid])))
            model_vintage = pd.DataFrame({'pass_rate': model_x, 'cum_vintage': model_y, 'model': str(i + 1)})
            result_vintage.append(model_vintage)

        result_vintage = pd.concat(result_vintage)

        print(ggplot(data = result_vintage[result_vintage['pass_rate'] >= p], mapping = aes(x = 'pass_rate')) #[result_vintage['pass_rate'] > ]
              + geom_line(aes(y = 'cum_vintage', color = 'model')) #, alpha = 0.8
              + labs(x = 'Pass Rate', y = 'Cumulative Bad Debt Rate', title = f'{month} ({result.shape[0]}): {title}')
              + scale_color_hue(name = 'Model Type', breaks = breaks_list, 
                           labels = model_name_list)
            )
        
        

'''
4. heatmap
'''
def get_heatmap(data: pd.core.frame.DataFrame, col_list: list, label: str, group_name: str, bins: int):
    '''
    计算各组里指定两列值的heatmap，每组内部等频分箱
    
    Parameters
    ----------
    
    data: 包含两列值、label列
    col_list：两列值
    label: 计算风险的指标
    group_name：组名
    bins：单列等频分箱数
    return：
    '''
    
    result = []
    for col in col_list:
#         data[col + '_qbin'] = pd.qcut(data[col], bins)
        data[col + '_qbin_group'] = pd.qcut(data[col], bins, labels = False) + 1
        data[col + '_qbin_group'] = data[col + '_qbin_group'].astype(str)
        
    heatmap_df = data.groupby([col + '_qbin_group' for col in col_list]).agg(
        {
            label: ['count', 'sum']
        }
    ).reset_index()

    heatmap_df.columns = [col + '_qbin' for col in col_list] + ['count', f'{label}_cnt']

    sum_df1 = heatmap_df.groupby(col_list[0] + '_qbin').agg(
        {
            'count': 'sum',
            f'{label}_cnt': 'sum'
        }
    ).reset_index()
    sum_df1.columns = [col_list[0] + '_qbin', 'count', f'{label}_cnt']
    sum_df1[col_list[1] + '_qbin'] = 'overall'

    sum_df2 = heatmap_df.groupby(col_list[1] + '_qbin').agg(
        {
            'count': 'sum',
             f'{label}_cnt': 'sum'
        }
    ).reset_index()
    
    sum_df2.columns = [col_list[1] + '_qbin', 'count', f'{label}_cnt']
    sum_df2[col_list[0] + '_qbin'] = 'overall'

    heatmap_df = pd.concat([heatmap_df, sum_df1, sum_df2]).reset_index(drop = True)

    heatmap_df['count_ratio'] = heatmap_df['count'] / data.shape[0]
    heatmap_df[f'{label}_ratio'] = heatmap_df[f'{label}_cnt'] / heatmap_df['count']
    
    heatmap_df['lift'] = heatmap_df[f'{label}_ratio'] / data[label].mean() #
    
    heatmap_df['count_ratio_%'] = heatmap_df['count_ratio'].apply(lambda x: str(round(x * 100, 1)) + '%')
    heatmap_df[f'{label}_ratio_%'] = heatmap_df[f'{label}_ratio'].apply(lambda x: str(round(x * 100, 1)) + '%')
    
    heatmap_df[f'lift_%'] = heatmap_df['lift'].apply(lambda x: str(round(x, 2)))

    # 调整下显示的颜色幅度
    heatmap_df['count_%'] = heatmap_df['count']
    heatmap_df[f'{label}_ratio'] = heatmap_df[f'{label}_ratio'] / 10
    heatmap_df['count_ratio'] = heatmap_df['count_ratio'] / 10
    heatmap_df['count'] = heatmap_df['count_ratio']
    heatmap_df['lift'] = heatmap_df['lift'] / 100

    for fill_ in ['count', 'count_ratio', f'{label}_ratio', 'lift']:
        df = heatmap_df[[col_list[0] + '_qbin', col_list[1] + '_qbin', fill_, f'{fill_}_%']]
        df.columns = [col_list[0] + '_qbin', col_list[1] + '_qbin', 'fill', 'fill_%']
        df['type'] = fill_
        df['group'] = group_name

        result.append(df)

    result = pd.concat(result).reset_index(drop = True)

    plotnine.options.figure_size = (16, 4)
    
    p = (ggplot(data = result, mapping = aes(col_list[0] + '_qbin', col_list[1] + '_qbin', fill = 'fill'))
         + geom_tile(aes(fill = 'fill'), colour = 'white')
         + scale_fill_gradient(name = "Value", low = "white", high = "red")
         + geom_text(aes(label = 'fill_%'), size = 8)
#          + labs(title = f'{m} Indexs{i} Overdue Ratio')
         + facet_grid('group ~ type')
     )
    
    print(p)
    
    return p, result



'''
5. 分组，组内按model_list等频分箱，箱内均值
'''
def get_group_bins_targets_mean(data: pd.core.frame.DataFrame, model_list: list, col_list: list, group_name: str, bins: int = 10):
    '''
    计算各组里，按model_list等频分箱，箱内指定指标均值
    
    Parameters
    ----------
    
    data: 包含组别（group_name），各指标
    model_list: 
    col_list: 需要计算均值的指标list
    group_name: 组名
    bins: 单列等频分箱数
    return: 返回的‘指标_{i}’为model_list[i]的指标均值
    '''
    
    grp_list = sorted(data[group_name].unique())
    
    rs_df = []
    
    for grp in grp_list:
        df = data[
            data[group_name] == grp
        ].reset_index(drop = True)
        
        for i, col in enumerate(model_list):
            df[col + '_group'] = pd.qcut(df[col], bins, labels = False) + 1

            rs1 = df.groupby(col + '_group').agg(
                {
                    col: 'mean' for col in col_list
                }
            ).reset_index()
            rs1.rename(
                columns = {
                    col + '_group': 'group'
                },
                inplace = True
            )
            rs1.rename(
                columns = {
                    col: col.split('___', 1)[-1] + f'_{i}' for col in col_list
                },
                inplace = True
            )

            if i == 0:
                rs = rs1.copy()

            else:
                rs = pd.merge(
                    rs,
                    rs1,
                    how = 'outer',
                    on = 'group'
                ).sort_values(by = 'group').reset_index(drop = True)

        rs[group_name] = grp

        rs_df.append(rs[
            [
                group_name, 'group'
            ] + list(
                chain.from_iterable([
                    [col.split('___', 1)[-1] + f'_{i}' for i in range(len(model_list))] for col in col_list
                ])
            )
        ])

    rs_df = pd.concat(rs_df).reset_index(drop = True)
    
    return rs_df



'''
6. 特征分布、lift、SHAP-value汇总 python
'''
def get_feature_analysis_plot(
    comp_data: pd.core.frame.DataFrame, 
    label_list: list, 
    col_list: list, 
    col_group_mapping: dict, 
    group_name: str, 
    model_feat_list: list, 
    tree_explainer_res: shap._explanation.Explanation, 
    feature_meanings: dict = {}
):
    '''
    按col_list分箱，计算各组内，分箱特征的分布、风险lift、SHAP-value
    
    Parameters
    ----------
    
    data: 包含组别（group_name），各分箱结果，各labels
    label_list: 需要计算lift的各labels
    col_list: 需要分析的特征list
    col_group_mapping: 需要分析的特征对应的分箱列名
    group_name: 组名
    model_feat_list: 入模特征list，保序，用于SHAP-value的输出
    tree_explainer_res: shap._explanation.Explanation
    feature_meanings: dict:{col: col的中文含义}，默认传{}则显示英文
    return: 
    '''
    size = 12
    myfont = fm.FontProperties(fname = '/data/public_data/cpu1/jiahou/SimHei.ttf', size = size) 
    
    plot_length = len(col_list)
    
    fig, axes = plt.subplots(plot_length, 2 + len(label_list), figsize = (40, 7 * plot_length))
    plt.yticks(rotation = 45)
    
    grp_list = list(comp_data[group_name].unique())
    if np.nan in grp_list:
        grp_list.remove(np.nan)
        
    grp_list.sort()
    
    for idx, feat in enumerate(col_list):
        
        if feature_meanings == {}:
            feat_meaning = feat
            
        else:
            feat_meaning = feature_meanings[feat]
        
        # 样本分箱占比
        res_dis = []

        for i, grp in enumerate(grp_list):
            data_i = comp_data[
                comp_data[group_name] == grp
            ]
            
            rs = data_i.groupby(col_group_mapping[feat]).agg(
                {'trace_id': 'count'}
            ) / data_i.shape[0]
            
            res_dis.append(rs)
            
        res_df_dis = pd.concat(res_dis, axis = 1)
        res_df_dis.fillna(0, inplace = True)
        res_df_dis.columns = grp_list
        
        res_df_dis_normalized = res_df_dis.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis = 0)
        
        # label_list lift
        res_label_dict = {}
        
        for label in label_list:
            res_label_dict[label] = {}
            
            res = []
            for i, grp in enumerate(grp_list):
                data_i = comp_data[
                    comp_data[group_name] == grp
                ]
                
                rs = data_i.groupby(col_group_mapping[feat]).agg(
                    {label: 'mean'}
                ) / data_i[label].mean()
                res.append(rs)
                
            res_df = pd.concat(res, axis = 1)
            res_df.columns = grp_list
            
            res_df_normalized = res_df.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis = 0)
            
            res_label_dict[label]['lift'] = res_df
            res_label_dict[label]['lift_norm'] = res_df_normalized
            
        # 作图
        ## 分布
        annot_dis = res_df_dis.mul(100).round(1).astype('str').add('%')
        ax1 = sns.heatmap(
            res_df_dis_normalized, 
            annot = annot_dis, 
            fmt = "", 
            cmap = 'Reds', 
            vmin = res_df_dis_normalized.min().min(), 
            vmax = res_df_dis_normalized.max().max(), 
            ax = axes[idx, 0] if plot_length > 1 else axes[0],  # ax = axes[idx, 0]
            annot_kws = {"fontsize": size}
        )
        # ax.invert_yaxis()
        ax1.set_title(f'{feat_meaning}', fontsize = size, fontproperties = myfont)
        ax1.tick_params(axis = 'both', labelsize = size, labelrotation = 45)
        ax1.set_xlabel('trace_dis', fontsize = size)
        ax1.set_ylabel('feat_bin', fontsize = size)
        # 将x轴坐标设置在上边
        ax1.xaxis.set_ticks_position('top')
        ax1.xaxis.set_label_position('top')
        
        ## label_list lift
        for i, label in enumerate(label_list):
            annot_label = res_label_dict[label]['lift'].round(2).astype('str')
            ax_i = sns.heatmap(
                res_label_dict[label]['lift_norm'],
                annot = annot_label,
                fmt = "", 
                cmap = 'Blues', 
                vmin = res_label_dict[label]['lift_norm'].min().min(), 
                vmax = res_label_dict[label]['lift_norm'].max().max(), 
                ax = axes[idx, i + 1] if plot_length > 1 else axes[i + 1], # ax = axes[idx, i + 1]
                annot_kws = {"fontsize": size}
            )
            
            ax_i.tick_params(axis = 'both', labelsize = size, labelrotation = 45)
            ax_i.set_xlabel(label, fontsize = size)
            ax_i.set_ylabel('feat_bin', fontsize = size)

            # 将x轴坐标设置在上边
            ax_i.xaxis.set_ticks_position('top')
            ax_i.xaxis.set_label_position('top')
            
        ## shap-value
        index_col = model_feat_list.index(feat)

        shap.plots.scatter(
            tree_explainer_res[:, index_col], 
            ax = axes[idx, len(label_list) + 1] if plot_length > 1 else axes[len(label_list) + 1], # ax = axes[idx, len(label_list) + 1]
            show = False
        )
        # ax2.set_ylabel('')
    
    plt.subplots_adjust(hspace = 0.5)

    plt.show()
    
    
    
'''
7. 逐月组间分布
'''
def get_group_dist_bymonth(
    data: pd.core.frame.DataFrame,
    group_name: str,
    month_name: str
):
    '''
    计算每个月份内，各组的占比
    
    Parameters
    ----------
    
    data: 包含trace_id，组别（group_name），月份字段（month_name）
    group_name:
    month_name: 月份字段
    return: 
    '''
    rs = []
    month_list = sorted(list(data[month_name].unique()))
    
    for month in month_list:
        df = data[
            (data[month_name] == month)
        ][['trace_id', group_name]]

        dff = df.groupby(group_name).agg(
            {
                'trace_id': 'count'
            }
        )
        dff.columns = ['cnt']
        dff['cnt_ratio'] = dff['cnt'] / dff['cnt'].sum()

        dff.rename(
            columns = {
                col: f'{col}_{month}' for col in dff.columns
            },
            inplace = True
        )

        rs.append(dff)

    rs = pd.concat(rs, axis = 1)
    rs = rs[
        ['cnt_' + month for month in month_list] +\
        ['cnt_ratio_' + month for month in month_list]
    ]
    
    return rs



'''
8. 模型分箱逐月指标刻画
'''
def get_model_target_info(
    df: pd.core.frame.DataFrame,
    model_bin: str,
    group: str,
    col_list: list,
    need_lift: list
):
    '''
    计算每个group（月份）内，各bin样本的指标均值，lift
    
    Parameters
    ----------
    
    df: 包含trace_id，模型分箱（model_bin），月份字段（group），指标list
    model_bin: 模型分箱
    group: 逐group统计，如 月份字段
    col_list: 需要计算均值的指标list
    need_lift: col_list中需要计算lift的指标
    return: 
    '''
    group_list = sorted(list(df[group].unique()))
    
    rs = []
    for grp in group_list:
        df_i = df[
            df[group] == grp
        ].reset_index(drop = True)
        
        dff_1 = df_i.groupby(model_bin).agg(
            {
                'trace_id': 'count'
            }
        ).reset_index()
        dff_1.rename(
            columns = {
                'trace_id': 'count'
            },
            inplace = True
        )
        dff_1['cnt_ratio'] = dff_1['count'] / dff_1['count'].sum()
        
        dff_2 = df_i.groupby(model_bin).agg(
            {
                col: 'mean' for col in col_list
            }
        ).reset_index()
        dff_2.rename(
            columns = {
                col: col + '_mean' for col in col_list
            },
            inplace = True
        )
        
        for col in need_lift:
            dff_2[col + '_mean_lift'] = dff_2[col + '_mean'] / df_i[col].mean()
        
        
        rs_i = pd.merge(
            dff_1,
            dff_2,
            how = 'outer',
            on = model_bin
        ).reset_index(drop = True)
        
        rs_i[group] = grp
        
        rs.append(rs_i)
        
    rs = pd.concat(rs).reset_index(drop = True)
    rs = rs[
        [
            group, model_bin, 'count', 'cnt_ratio'
        ] + [
            col + '_mean' for col in col_list
        ] + [
            col + '_mean_lift' for col in need_lift
        ]
    ]
        
    return rs