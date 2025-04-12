# 宏观预测code version 3
# PCA+ARIMA预测基准场景
# 假设预测的分布和历史的分布是一致的
# 采用加减每个指标10Y标准差的方式进行极端场景的生成


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
import os
import pmdarima as pm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import scipy.stats
import warnings
import statsmodels.stats.diagnostic as smd
warnings.filterwarnings("ignore")


# =======================================================================
# # 关键参数选取
# =============================================================================

# main components variance threshold（可解释的variance来选取主成分个数）
variance_threshold = 0.8
# 单变量筛选 pvalue r-squared
p_value_threshold = 0.1
r2_threshold = 0.05
# 预测周期数
predict_period = 9

n_normal = 1 # 悲观、乐观scenario = n个标准差
n_extreme = 3 # 极度悲观、乐观scenario = n个标准差


min_score = 0
max_score = 100
# 入PCA数据范围
date_s  = '2015-12-31'


# =======================================================================
# # Read Input Data
# =============================================================================

working_dir =  r'E:\PCA' #<---更新代码路径#

macro_data = pd.read_excel(r"{}\Input_Data_2023.6.xlsx".format(working_dir), sheet_name="Macro_Data") #<---更新输入文件名#
# SUBSET data
macro_data_all = macro_data[macro_data['指标名称']>=date_s]
macro_data_10Y = macro_data[macro_data['指标名称']>='2015-12-31']


macro_list = pd.read_excel(r"{}\Input_Data_恒丰对公整体PD_2023.9.xlsx".format(working_dir), sheet_name="Code") #<---更新输入文件名#
macro_list_candidates =  list(macro_list[macro_list['是否入模']=='Y']['指标名称'])
macro_list_candidates_code =  list(macro_list[macro_list['是否入模']=='Y']['指标名称'])




mapping_dict = {}
for i in range(len(macro_list_candidates)):
    mapping_dict[macro_list_candidates[i]] = macro_list_candidates_code[i]

macro_data_all = macro_data_all[macro_list_candidates]
macro_data_10Y = macro_data_10Y[macro_list_candidates]

macro_list_sign =  macro_list[macro_list['是否入模']=='Y'][['指标名称','sign']].set_index('指标名称')



# 缺失值线性插值filled
macro_data_filled = macro_data_all.interpolate(method='linear',axis=0)
macro_data_10Y_filled = macro_data_10Y.interpolate(method='linear',axis=0)

# 标准化
# 历史均值
Mean = np.mean(macro_data_filled)
# 历史标准值
Std = np.std(macro_data_filled, ddof=1)
# 十年标准差
Std_10Y = np.std(macro_data_10Y_filled, ddof=1)
macro_data_normalized=(macro_data_filled-Mean)/(Std)



# =============================================================================
# n_components是主成分个数,画出来方差贡献度，看看几个成分就可以解释出来了
# 先跑一轮PCA循环，看看几个因子解释力度够用
# =============================================================================
pca_model = PCA(n_components=min(len(macro_data_normalized.columns),len(macro_data_normalized)))
pca_model.fit(macro_data_normalized)
variance_cumsum = np.cumsum(pca_model.explained_variance_ratio_)
num_pc = next(x for x, val in enumerate(variance_cumsum) if val >= variance_threshold)+1




# =============================================================================
# #pca_train = PCA(n_components=c) 中的C是要保留的主成分的个数
# num_pc个主成分
# =============================================================================
pca_train = PCA(n_components=num_pc)
pca_result = pca_train.fit_transform(macro_data_normalized)
df_pca_result  = pd.DataFrame(pca_result,index=macro_data_normalized.index)

# 单变量筛选

univariace_info = []
for tmp_marco_var in macro_data_normalized.columns:

    tmp_dep = macro_data_normalized[tmp_marco_var]

    for i in range(num_pc):
        tmp_ind = df_pca_result[i]
        X = sm.add_constant(tmp_ind)
        model = sm.OLS(tmp_dep,X)
        LR = model.fit()
        LR_R2 = LR.rsquared
        LR_R2_adj = LR.rsquared_adj
        LR_Pvalue = LR.pvalues.values[1]
        univariace_info.append({'marcro_variable':tmp_marco_var,
                                'pca_component':i,
                                'r_squared':LR_R2,
                                'p_value':LR_Pvalue})
        
df_univariace_info = pd.DataFrame(univariace_info)
    
# 筛选
df_univariace_info_filter_p = df_univariace_info[df_univariace_info['p_value']<p_value_threshold]
df_univariace_info_filter = df_univariace_info_filter_p[df_univariace_info_filter_p['r_squared']>r2_threshold]


# significant PC regression
coef_total = []
p_value_total = []
ind_name_total = []
dep_name_total = []
r_squared_total=[]
reset_test = []
DW_test= []
sharp_test = []
BP_test = []
VIF_test=[]    



for tmp_marco_var in macro_data_normalized.columns:
    tmp_ind_variable_list = df_univariace_info_filter.loc[df_univariace_info_filter['marcro_variable']==tmp_marco_var,'pca_component']
    tmp_ind_variable_list_new = ['Intercept'] +tmp_ind_variable_list.tolist()
    tmp_ind = df_pca_result[tmp_ind_variable_list]
    tmp_x = sm.add_constant(tmp_ind)
    tmp_y = macro_data_normalized[tmp_marco_var]
    tmp_model = sm.OLS(tmp_y,tmp_x)
    tmp_LR = tmp_model.fit()
    tmp_LR_R2 = tmp_LR.rsquared  
    tmp_LR_R2_adj = tmp_LR.rsquared_adj
    tmp_LR_Pvalue = tmp_LR.pvalues
    tmp_LR_coef = tmp_LR.params
    
    from statsmodels.stats.stattools import durbin_watson
    from scipy.stats import shapiro
    import statsmodels.stats.api as sms
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    resettest = smd.linear_reset(res=tmp_LR, power=2, test_type="fitted", use_f=True)
    reset_pvalue = resettest.pvalue
    dw_pvalue = durbin_watson(tmp_LR.resid)
    shapiro_pvalue = shapiro(tmp_LR.resid)[1]
    test = sms.het_breuschpagan(tmp_LR.resid, tmp_LR.model.exog)
    viftest_list = [variance_inflation_factor(tmp_x.values, i) for i in range(tmp_x.shape[1])]

    dep_name_total.extend([tmp_marco_var]*len(tmp_ind_variable_list_new))
    ind_name_total.extend(tmp_ind_variable_list_new)
    coef_total.extend(tmp_LR_coef)
    p_value_total.extend(tmp_LR_Pvalue)
    r_squared_total.extend([tmp_LR_R2] * len(tmp_ind_variable_list_new))
    
    reset_test.extend([reset_pvalue]* len(tmp_ind_variable_list_new))
    DW_test.extend([dw_pvalue]* len(tmp_ind_variable_list_new))
    sharp_test.extend([shapiro_pvalue]* len(tmp_ind_variable_list_new))
    BP_test.extend([test[1]]* len(tmp_ind_variable_list_new))
    VIF_test.extend(viftest_list)
    
# build dataframe
df_pca_regression = pd.DataFrame({'macro_variable':dep_name_total,
                                  'principal':ind_name_total,
                                  'coef':coef_total,
                                  'p_value':p_value_total,
                                  'r_squared':r_squared_total,
                                  'reset_test':reset_test,
                                  'DW_test':DW_test,
                                  'sharp_test':sharp_test,
                                  'BP_test':BP_test,
                                  'VIF_test': VIF_test
                                  }) 
                                  
                              


# =============================================================================
# 对PC做AUTO ARIMA预测（只有基准场景）
# =============================================================================

# initialized
# 基准
df_pca_result_predict_base = pd.DataFrame()


for i in range(num_pc):
    tmp_pc = df_pca_result[i]
    
    tmp_model_auto = pm.auto_arima(tmp_pc, start_p=1, start_q=1,
                               information_criterion='bic', # aic or bic
                               test='kpss',### adf or kpss
                               max_p=5, max_q=5,
                               m=4, ### seasonality
                               d=None,
                               seasonal=True,
                               start_s=0,
                               D=0,
                               trace=True,
                               error_action='ignore',
                               suppress_warnings=True,
                               Stepwise=False)
    
    tmp_pc_forecast_auto = tmp_model_auto.predict(n_periods=predict_period)
    print(tmp_model_auto.summary())
   
    # convert into dataframe
    tmp_pc_forecast_auto_base = pd.DataFrame(tmp_pc_forecast_auto,columns=[i])
    # append into one
    df_pca_result_predict_base = pd.concat([df_pca_result_predict_base,tmp_pc_forecast_auto_base],axis=1)
    
# predict的日期list
predict_list = list(df_pca_result_predict_base.index)

df_pca_regression_predict_base = df_pca_regression

sum_dict = {}
for tmp_predict in predict_list:
    sum_dict[str(tmp_predict)+'_predict'] = sum
    # obtain perdict value under diff scenarios
    tmp_predict_pca_base = df_pca_result_predict_base[df_pca_result_predict_base.index == tmp_predict].T.reset_index().rename(columns = {'index':'principal'})
    df_pca_regression_predict_base = pd.merge(df_pca_regression_predict_base,tmp_predict_pca_base,left_on= 'principal',right_on= 'principal',how = 'left')

    # filling intercept
    df_pca_regression_predict_base.fillna(1,inplace = True)


    df_pca_regression_predict_base[str(tmp_predict)+'_predict'] = df_pca_regression_predict_base['coef']*df_pca_regression_predict_base[tmp_predict]

df_pca_result_base = df_pca_regression_predict_base.groupby(by='macro_variable').agg(sum_dict)

# de-normalized
predict_list_new = df_pca_result_base.columns
# mean
df_Mean = pd.DataFrame(Mean,columns=['mean'])
# std
df_Std = pd.DataFrame(Std,columns=['std'])
# 10Y std
df_Std_10Y = pd.DataFrame(Std_10Y,columns=['std_10Y'])


# 历史最大
df_max = pd.DataFrame(np.max(macro_data_filled),columns=['max'])
# 历史最小
df_min = pd.DataFrame(np.min(macro_data_filled),columns=['min'])

# 10y最大
df_max_10Y = pd.DataFrame(np.max(macro_data_10Y_filled),columns=['max_10Y'])
# 10y最小
df_min_10Y = pd.DataFrame(np.min(macro_data_10Y_filled),columns=['min_10Y'])


#df_pca_result_base = pd.concat([df_pca_result_base,df_Mean,df_Std])

df_pca_result_base = pd.concat([df_pca_result_base,df_Mean,df_Std,df_Std_10Y],axis=1)


# 最终基准预测
df_pca_regression_predict_final_base = pd.DataFrame()
for tmp_predict in predict_list_new:
    tmp_de_normalize_base = df_pca_result_base[tmp_predict]*df_pca_result_base['std']+df_pca_result_base['mean']
    tmp_de_normalize_base.name = tmp_predict
    
    df_pca_regression_predict_final_base = pd.concat([df_pca_regression_predict_final_base,tmp_de_normalize_base],axis=1)


df_pca_regression_predict_final = pd.concat([df_pca_regression_predict_final_base,df_Std,df_Std_10Y,df_max,df_min,df_max_10Y,df_min_10Y,macro_list_sign],axis = 1)



df_pca_regression_predict_final_opt = pd.DataFrame()
df_pca_regression_predict_final_pes = pd.DataFrame()
'''
df_pca_regression_predict_final_ext_opt = pd.DataFrame()
df_pca_regression_predict_final_ext_pes = pd.DataFrame()
'''


for tmp_period in df_pca_regression_predict_final_base.columns:
    
    # 乐观
    df_pca_regression_predict_final_opt[tmp_period] = df_pca_regression_predict_final[tmp_period]+n_normal*df_pca_regression_predict_final['std_10Y']*df_pca_regression_predict_final['sign']
    # 悲观
    df_pca_regression_predict_final_pes[tmp_period] = df_pca_regression_predict_final[tmp_period]-n_normal*df_pca_regression_predict_final['std_10Y']*df_pca_regression_predict_final['sign']
''' 
    # 极度乐观
    df_pca_regression_predict_final_ext_opt[tmp_period] = df_pca_regression_predict_final[tmp_period]+n_extreme*df_pca_regression_predict_final['std_10Y']*df_pca_regression_predict_final['sign']
    # 极度悲观
    df_pca_regression_predict_final_ext_pes[tmp_period] = df_pca_regression_predict_final[tmp_period]-n_extreme*df_pca_regression_predict_final['std_10Y']*df_pca_regression_predict_final['sign']
  '''  



# output
with pd.ExcelWriter(r"{}\macro_data_predict.xlsx".format(working_dir)) as writer:
    df_pca_regression_predict_final_base = pd.concat([macro_data_filled,df_pca_regression_predict_final_base.T])
    df_pca_regression_predict_final_base.rename(columns = mapping_dict,inplace = True)
    df_pca_regression_predict_final_base.to_excel(writer, sheet_name="基准")
    
    df_pca_regression_predict_final_opt = pd.concat([macro_data_filled,df_pca_regression_predict_final_opt.T])
    df_pca_regression_predict_final_opt.rename(columns = mapping_dict,inplace = True)
    df_pca_regression_predict_final_opt.to_excel(writer, sheet_name="乐观")
    
    df_pca_regression_predict_final_pes = pd.concat([macro_data_filled,df_pca_regression_predict_final_pes.T])
    df_pca_regression_predict_final_pes.rename(columns = mapping_dict,inplace = True)
    df_pca_regression_predict_final_pes.to_excel(writer, sheet_name="悲观")
    '''   df_pca_regression_predict_final_ext_opt = pd.concat([macro_data_filled,df_pca_regression_predict_final_ext_opt.T])
    df_pca_regression_predict_final_ext_opt.rename(columns = mapping_dict,inplace = True)
    df_pca_regression_predict_final_ext_opt.to_excel(writer, sheet_name="极度乐观")
    
    df_pca_regression_predict_final_ext_pes = pd.concat([macro_data_filled,df_pca_regression_predict_final_ext_pes.T])
    df_pca_regression_predict_final_ext_pes.rename(columns = mapping_dict,inplace = True)
    df_pca_regression_predict_final_ext_pes.to_excel(writer, sheet_name="极度悲观")
 '''   
    df_pca_regression.to_excel(writer, sheet_name="pca regression")
    
    df_pca_result.to_excel(writer, sheet_name="pca 主成分历史值")
    
    df_pca_result_base.to_excel(writer, sheet_name="pca 主成分基准预测值")
    
    df_pca_result_predict_base.to_excel(writer, sheet_name="pca 基准预测值")