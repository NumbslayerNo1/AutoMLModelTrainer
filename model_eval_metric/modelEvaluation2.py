# -*- coding: utf8  -*-

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn import metrics

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import math
import copy
import sys
sys.path.append('/data1/mex_reloan_data/code_')  # 将package所在目录加入执行目录列表
import modelEvaluation
import psiCalculation

'''
本评估代码包括：

1、累积逾期曲线 get_cumulative_bad_debt_rate_plot()

2、分bin-lift画像等统计：cal_stats_part()

3、模型分布图示版：get_model_distribution

4、模型分布分bin比例版：distribution_bin

5、交叉评估heatmap: cross_heatmap


'''



'''
累积逾期曲线
'''

def get_cumulative_bad_debt_rate_plot(df: pd.core.frame.DataFrame, # 评估数据集
                                      term: int, # label期数
                                      ipd7: str, # 计算AUC的label
                                      unpaid: str, # 逾期曲线分子
                                      principal: str, # 逾期曲线分母
                                      title: str, # 标题
                                      month: str, # 月份名称
                                      score_list: list, # model score_list
                                      model_name_list: list, # model_name_list
                                      p = 0. # 剔除曲线头部比例
                                     ):
    result = df
    result.reset_index(drop = True, inplace = True)
    
    model_name_list_cp = model_name_list.copy()
    
    # 计算AUC
    for i in range(len(score_list)):
        auc_pd7 = metrics.roc_auc_score(result[ipd7], result[score_list[i]])
        
        model_name_list[i] = f'{model_name_list[i]} | {term}pd7-AUC {round(auc_pd7, 4)}'
    
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
        model_vintage = pd.DataFrame({'pass_rate': model_x, 'cum_vintage': model_y, 'model': model_name_list[i]})
        result_vintage.append(model_vintage)
        
    result_vintage = pd.concat(result_vintage)
#     display(result_vintage)
    
    # 设置绘图大小
    figure_size = (16, 8)
    plt.figure(figsize=figure_size)
    sns.set_style('darkgrid')
    # 使用 seaborn 绘制点线图

    result_vintage = result_vintage[result_vintage['pass_rate'] >= p]
    result_vintage = result_vintage.reset_index()

    sns.lineplot(data=result_vintage, x='pass_rate', y='cum_vintage', hue='model', palette='tab10')

    # 设置标签和标题
    plt.xlabel('pass_rate')
    plt.ylabel('cum_vintage')
    plt.title(f'{month}: {title}')

    # 增加图例并设置图例位置和名称
    plt.legend(title='Model Type', loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

    # plt.tight_layout()
    # 显示图形
    plt.show()
    
    
    
'''
分bin统计人头风险均值、Lift、下单率、额度使用率、风控前后授信额度、历史最大逾期天数、在贷笔数等统计画像
'''    
def calc_auc(y_prob, y_label):
    fpr, tpr, _ = roc_curve(y_label, y_prob)
    return auc(fpr, tpr)


def calc_ks(y_prob, y_label):
    fpr, tpr, _ = roc_curve(y_label, y_prob)
    return np.max(tpr - fpr)


def cal_stats_part(df: pd.core.frame.DataFrame, # 评估数据集
                   model_score: str, # 模型名称
                   basic_info: dict, # 基础字段统计dict
                   label: str, # 风险标签
                   index_dic: dict, # 画像统计dict
                   q = 10 # 模型分bin数量
                  ):

    df = df.copy()
    df[f'{model_score}_group'] = pd.qcut(df[model_score], q)
    group_col = f"{model_score}_group"

    df["good"] = 1 - df[label]
    df["bad"] = df[label]

    group = df.groupby(group_col)

    # -----------------------
    # 基础统计
    # -----------------------
    res = group[label].agg(
        num="count",
        bad_num="sum"
    ).reset_index()

    res["good_num"] = res["num"] - res["bad_num"]

    # prob range
    prob_range = group[model_score].agg(["min", "max"]).reset_index()
    prob_range.columns = [group_col, "prob_min", "prob_max"]

    res = res.merge(prob_range, on=group_col)

    # -----------------------
    # 业务指标
    # -----------------------
    basic_df = group.agg(basic_info).reset_index()
    index_df = group.agg(index_dic).reset_index()

    res = res.merge(basic_df, on=group_col)
    res = res.merge(index_df, on=group_col)

    # -----------------------
    # bin auc
    # -----------------------
    bin_auc = group.apply(
        lambda x: calc_auc(x[model_score], x[label])
    ).rename("bin_auc")

    res = res.merge(bin_auc, left_on=group_col, right_index=True)

    # -----------------------
    # 全局统计
    # -----------------------
    total_n = len(df)
    total_bad = df["bad"].sum()
    total_good = df["good"].sum()

    res["num_ratio"] = res["num"] / total_n
    res["bad_ratio"] = res["bad_num"] / res["num"]

    # -----------------------
    # 排序
    # -----------------------
    res = res.sort_values(group_col)

    res["good_cumsum"] = res["good_num"].cumsum() / total_good
    res["bad_cumsum"] = res["bad_num"].cumsum() / total_bad

    res["ks"] = np.abs(res["bad_cumsum"] - res["good_cumsum"])

    # -----------------------
    # lift
    # -----------------------
    res_desc = res.iloc[::-1].copy()

    res_desc["inv_bad_cum"] = res_desc["bad_num"].cumsum() / total_bad
    res_desc["inv_num_cum"] = res_desc["num"].cumsum() / total_n

    res_desc["lift"] = res_desc["inv_bad_cum"] / res_desc["inv_num_cum"]

    avg_bad_rate = total_bad / total_n
    res_desc["bin_lift"] = res_desc["bad_ratio"] / avg_bad_rate

    res = res.merge(
        res_desc[[group_col, "lift", "bin_lift"]],
        on=group_col
    )

    # -----------------------
    # 累积AUC
    # -----------------------
    borders = res[group_col].apply(lambda x: x.left).values

    acc_auc = {}
    for b in borders:
        sub = df[df[group_col].apply(lambda x: x.left) <= b]
        acc_auc[b] = calc_auc(sub[model_score], sub[label])

    res["acc_auc"] = res[group_col].apply(lambda x: acc_auc[x.left])

    # -----------------------
    # bin label
    # -----------------------
    res["bin"] = res[group_col].apply(
        lambda x: f"({x.left:.4f}, {x.right:.4f}]"
    )

    # -----------------------
    # rounding
    # -----------------------
    round_cols = [
        "num_ratio", "bad_ratio", "good_cumsum",
        "bad_cumsum", "ks", "acc_auc",
        "bin_auc", "lift", "bin_lift"
    ]

    res[round_cols] = res[round_cols].round(4)

    # -----------------------
    # 输出列
    # -----------------------
    output_cols = (
        ["bin"]
        + list(basic_info.keys())
        + [
            "num_ratio",
            "good_num", "bad_num",
            "bad_ratio",
            "good_cumsum", "bad_cumsum",
            "ks",
            "acc_auc", "bin_auc", "bin_lift"
        ]
        + list(index_dic.keys())
    )

    return res[output_cols]

    
    

'''
获取模型分布 ： 图示版
'''

def get_model_distribution(
                           data: pd.core.frame.DataFrame, # 获取评估数据
                           time_group: str, # 模型分布时间分组
                           start_time_group: str, # 初始时间分组，以初始分组计算后续分组PSI，统计分布稳定性
                           model_list: list, # 模型list
                           time_group_list: list, # 模型分布时间分组list
                           n_cols: int = 3  # 默认每行3列
):

    n_models = len(model_list)

    # -----------------------
    # 自动计算行数
    # -----------------------
    n_rows = math.ceil(n_models / n_cols)

    # -----------------------
    # 创建画布
    # -----------------------
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 10, n_rows * 10)
    )

    # 保证 axes 一定是二维
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # -----------------------
    # 开始画图
    # -----------------------
    for idx, col in enumerate(model_list):

        i = idx // n_cols
        j = idx % n_cols

        ax = axes[i, j]

        # base分布
        l_base = data.loc[
            data[time_group] == start_time_group, col
        ].dropna().values

        # 各时间分布
        for time_g in time_group_list:

            df = data.loc[
                data[time_group] == time_g, col
            ].dropna()

            if len(df) == 0:
                continue

            l_test = df.values

            psi = psiCalculation.get_psi(l_base, l_test)

            sns.kdeplot(
                df,
                label=f'{time_g} n={len(df)} | psi={psi:.6f}',
                ax=ax
            )

        ax.set_title(col)
        ax.legend()

    # -----------------------
    # 删除多余子图（关键优化）
    # -----------------------
    total_plots = n_rows * n_cols
    for idx in range(n_models, total_plots):
        i = idx // n_cols
        j = idx % n_cols
        fig.delaxes(axes[i, j])

    plt.tight_layout()
    plt.show()
    
    
    
'''
获取模型分布 ： 分bin比例版
'''

def distribution_bin(df: pd.core.frame.DataFrame, # 获取评估数据
                    time_group: str, # 模型分布时间分组
                    start_time_group: str, # 初始时间分组，以初始分组计算后续分组PSI，统计分布稳定性
                    model: str, # 模型分
                    time_group_list: list, # 模型分布时间分组list
                     q = 10 # 模型分bin数量
                    ):
    dis = pd.DataFrame()
    
    df_test = df[df[time_group] == start_time_group]
    df_test.loc[:, f'{model}_bin'], bins = pd.qcut(df_test[model], q=q, retbins=True, duplicates='drop')
    df.loc[:, f'{model}_bin'] = pd.cut(df[model], bins)
    
    for m in time_group_list:
        df_m = df[df[time_group] == m]
        dis.loc[0:(q - 1), f"{m} 一次风控count"] = df_m.groupby(f'{model}_bin').trace_id.count().values
        dis.loc[q, f"{m} 一次风控count"] = ''
    for m in time_group_list:
        df_m = df[df[time_group] == m]
        dis.loc[0:(q - 1), f"{m} 一次风控rate"] = (df_m.groupby(f'{model}_bin').trace_id.count()).values/len(df_m)
        psi = psiCalculation.get_psi(df_test[model].tolist(), df_m[model].tolist()) # 计算基于l_base的psi
        dis.loc[q, f"{m} 一次风控rate"] = psi
    
    dis.index = list(range(q)) + ['psi']
        
    display(dis)
    
    
    
'''
heatmap 结构说明

第一排：依次为1pd7 人头风险均值、当月最长可看期数（例如9月可看6pd7）人头风险均值，最长可看期数风险Lift、订单量、订单占比
第二排：依次为件均、风控前bf_credit、风控后af_credit、单笔额度使用率均值、额度使用率（sum(principal)/sum(order_credit)）均值

'''

def cross_heatmap(cross_data: pd.core.frame.DataFrame,  # heatmap data
                  x_score: str, # 横轴模型分名称
                  y_score: str, # 纵轴模型分名称
                  x_bins: str, # 横轴模型分bins名称
                  y_bins: str, # 纵轴模型分bins名称
                  maximum_observable_term_risk_label: str, # 时间分组内最大可看期数风险标签
                  term1_risk_label: str, # 时间分组内1期数风险标签
                  pred: str, # 标题模型名称
                  credit_usage_rate: str, # 额度使用率
                  order_principal: str, # 借款金额
                  order_credit: str, # 下单时可借金额
                  bf_credit: str, # 风控前授信额度
                  af_credit: str, # 风控后授信额度
                  bins: int = 5 # 默认模型分5bin交叉
                 ):
    
    # 获取模型分分bin
    cross_data.loc[:, x_bins] = pd.qcut(cross_data[x_score], q = bins)
    cross_data.loc[:, y_bins] = pd.qcut(cross_data[y_score], q = bins)
    
    fig, axes = plt.subplots(2, 5, figsize = (60, 2*7))
    
    fig.autofmt_xdate(rotation=45)
    
    plt.yticks(rotation=45)
    
    cross_data.loc[:, x_bins] = cross_data.loc[:, x_bins].apply(lambda interval: pd.Interval(round(interval.left, 3), interval.right))
    cross_data.loc[:, y_bins] = cross_data.loc[:, y_bins].apply(lambda interval: pd.Interval(round(interval.left, 3), interval.right))
    
    columns = cross_data.columns

    all_data1 = copy.deepcopy(cross_data)
    all_data1[y_bins] = 'OverAll'
    all_data1[x_bins] = cross_data[x_bins]

    all_data2 = copy.deepcopy(cross_data)
    all_data2[y_bins] = cross_data[y_bins]
    all_data2[x_bins] = 'OverAll'

    all_data3 = copy.deepcopy(cross_data)
    all_data3[y_bins] = 'OverAll'
    all_data3[x_bins] = 'OverAll'

    cross_data = pd.concat([cross_data, all_data1[columns], all_data2[columns], all_data3[columns]])
    
    stat_target_total_mean = np.array(cross_data.groupby([y_bins])[maximum_observable_term_risk_label].mean()).T
    
#     print(stat_target_total_mean)

    cross_df_dis = cross_data.groupby([y_bins, x_bins])[maximum_observable_term_risk_label].count().unstack()

    cross_df_rate = cross_data.groupby([y_bins, x_bins])[maximum_observable_term_risk_label].mean().unstack()

    cross_df_rate_annot = cross_df_rate.mul(100).round(2).astype('str').add('%')
    
    
    cross_df_rate_fpd15 = cross_data.groupby([y_bins, x_bins])[term1_risk_label].mean().unstack()

    cross_df_rate_fpd15_annot = cross_df_rate_fpd15.mul(100).round(2).astype('str').add('%')
    
#     print(cross_df_rate)
    
    cross_df_rate_lift = (cross_df_rate.T/stat_target_total_mean).T

    # annot
    cross_df_dis_rate = cross_data.groupby([y_bins, x_bins])[maximum_observable_term_risk_label].count().unstack()/all_data1.shape[0]
    cross_df_dis_rate_annot = cross_df_dis_rate.mul(100).round(1).astype('str').add('%')


    # 画像指标
    # 件均，bf_credit, af_credit, 额度使用率*2
    cross_df_principal = cross_data.groupby([y_bins, x_bins])[order_principal].mean().unstack()
    cross_df_bf_credit = cross_data.groupby([y_bins, x_bins])[bf_credit].mean().unstack()
    cross_df_af_credit = cross_data.groupby([y_bins, x_bins])[af_credit].mean().unstack()
    cross_df_credit_usage_rate_1 = cross_data.groupby([y_bins, x_bins])[credit_usage_rate].mean().unstack()
    cross_df_credit_usage_rate_2 = (cross_data.groupby([y_bins, x_bins])[order_principal].sum()/cross_data.groupby([y_bins, x_bins])[order_credit].sum()).unstack()
    
    # cross_df_principal,cross_df_bf_credit,cross_df_af_credit,cross_df_credit_usage_rate_1,cross_df_credit_usage_rate_2

    #cross_df
    
    ax = sns.heatmap(cross_df_rate_fpd15, annot=cross_df_rate_fpd15_annot, fmt="", cmap='Blues', vmin=cross_df_rate_fpd15.min().min(), vmax=cross_df_rate_fpd15.max().max(), ax = axes[0,0], 
                     annot_kws={"fontsize":20})
    ax.invert_yaxis()
    ax.set_title(f'{pred} | {term1_risk_label} proportion', fontsize=20)
    ax.tick_params(axis = 'both', labelsize = 20, labelrotation = 45)
    ax.set_xlabel(x_bins, fontsize=20)
    ax.set_ylabel(y_bins, fontsize=20)

    ax = sns.heatmap(cross_df_rate, annot=cross_df_rate_annot, fmt="", cmap='Blues', vmin=cross_df_rate.min().min(), vmax=cross_df_rate.max().max(), ax = axes[0, 1], 
                     annot_kws={"fontsize":20})
    ax.invert_yaxis()
    ax.set_title(f'{pred} | {maximum_observable_term_risk_label} proportion', fontsize=20)
    ax.tick_params(axis = 'both', labelsize = 20, labelrotation = 45)
    ax.set_xlabel(x_bins, fontsize=20)
    ax.set_ylabel(y_bins, fontsize=20)
    
    
    ax = sns.heatmap(cross_df_rate_lift, annot=True, fmt=".2f", cmap='Blues', vmin=cross_df_rate_lift.min().min(), vmax=cross_df_rate_lift.max().max(), ax = axes[0, 2], 
                     annot_kws={"fontsize":20})
    ax.invert_yaxis()
    ax.set_title(f'{pred} | {maximum_observable_term_risk_label} lift', fontsize=20)
    ax.tick_params(axis = 'both', labelsize = 20, labelrotation = 45)
    ax.set_xlabel(x_bins, fontsize=20)
    ax.set_ylabel(y_bins, fontsize=20)
    
    
    ax = sns.heatmap(cross_df_dis, annot=True, fmt=".0f", cmap='Blues', vmin=cross_df_dis.min().min(), vmax=cross_df_dis.max().max(), ax = axes[0, 3], 
                     annot_kws={"fontsize":20})
    ax.invert_yaxis()
    ax.set_title(f'{pred} | {maximum_observable_term_risk_label} count', fontsize=20)
    ax.tick_params(axis = 'both', labelsize = 20, labelrotation = 45)
    ax.set_xlabel(x_bins, fontsize=20)
    ax.set_ylabel(y_bins, fontsize=20)
    
    ax = sns.heatmap(cross_df_dis_rate, annot=cross_df_dis_rate_annot, fmt="", cmap='Blues', vmin=cross_df_dis_rate.min().min(), vmax=cross_df_dis_rate.max().max(), ax = axes[0, 4], 
                     annot_kws={"fontsize":20})
    ax.invert_yaxis()
    ax.set_title(f'{pred} | {maximum_observable_term_risk_label} count rate', fontsize=20)
    ax.tick_params(axis = 'both', labelsize = 20, labelrotation = 45)
    ax.set_xlabel(x_bins, fontsize=20)
    ax.set_ylabel(y_bins, fontsize=20)
    
    ## 画像指标 cross_df_principal,cross_df_bf_credit,cross_df_af_credit,cross_df_credit_usage_rate_1,cross_df_credit_usage_rate_2
    
    ax = sns.heatmap(cross_df_principal, annot=True, fmt=".0f", cmap='Blues', vmin=cross_df_principal.min().min(), vmax=cross_df_principal.max().max(), ax = axes[1, 0], 
                     annot_kws={"fontsize":20})
    ax.invert_yaxis()
    ax.set_title(f'{pred} | principal', fontsize=20)
    ax.tick_params(axis = 'both', labelsize = 20, labelrotation = 45)
    ax.set_xlabel(x_bins, fontsize=20)
    ax.set_ylabel(y_bins, fontsize=20)
    
    ax = sns.heatmap(cross_df_bf_credit, annot=True, fmt=".0f", cmap='Blues', vmin=cross_df_bf_credit.min().min(), vmax=cross_df_bf_credit.max().max(), ax = axes[1, 1], 
                     annot_kws={"fontsize":20})
    ax.invert_yaxis()
    ax.set_title(f'{pred} | bf_credit', fontsize=20)
    ax.tick_params(axis = 'both', labelsize = 20, labelrotation = 45)
    ax.set_xlabel(x_bins, fontsize=20)
    ax.set_ylabel(y_bins, fontsize=20)
    
    ax = sns.heatmap(cross_df_af_credit, annot=True, fmt=".0f", cmap='Blues', vmin=cross_df_af_credit.min().min(), vmax=cross_df_af_credit.max().max(), ax = axes[1, 2], 
                     annot_kws={"fontsize":20})
    ax.invert_yaxis()
    ax.set_title(f'{pred} | af_credit', fontsize=20)
    ax.tick_params(axis = 'both', labelsize = 20, labelrotation = 45)
    ax.set_xlabel(x_bins, fontsize=20)
    ax.set_ylabel(y_bins, fontsize=20)
    
    ax = sns.heatmap(cross_df_credit_usage_rate_1, annot=True, fmt=".2f", cmap='Blues', vmin=cross_df_credit_usage_rate_1.min().min(), vmax=cross_df_credit_usage_rate_1.max().max(), ax = axes[1, 3], 
                     annot_kws={"fontsize":20})
    ax.invert_yaxis()
    ax.set_title(f'{pred} | credit_usage_proportion', fontsize=20)
    ax.tick_params(axis = 'both', labelsize = 20, labelrotation = 45)
    ax.set_xlabel(x_bins, fontsize=20)
    ax.set_ylabel(y_bins, fontsize=20)

    ax = sns.heatmap(cross_df_credit_usage_rate_2, annot=True, fmt=".2f", cmap='Blues', vmin=cross_df_credit_usage_rate_2.min().min(), vmax=cross_df_credit_usage_rate_2.max().max(), ax = axes[1, 4], 
                     annot_kws={"fontsize":20})
    ax.invert_yaxis()
    ax.set_title(f'{pred} | credit_usage_proportion sum div version', fontsize=20)
    ax.tick_params(axis = 'both', labelsize = 20, labelrotation = 45)
    ax.set_xlabel(x_bins, fontsize=20)
    ax.set_ylabel(y_bins, fontsize=20)
    
    fig.tight_layout()
    plt.subplots_adjust(wspace = 0.1, hspace = 0.5)
    
    plt.show()
