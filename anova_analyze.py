# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 23:21:16 2024

@author: benker
"""


# 資料預處理
import warnings
warnings.filterwarnings("ignore")  # 忽略所有警告

import numpy as np
import pandas as pd
from statsmodels.multivariate.manova import MANOVA
from imblearn.over_sampling import RandomOverSampler
from scipy.stats import levene
from scipy.stats import f_oneway
data1 = pd.read_excel('C:\\Users\\benker\\Downloads\\112年度_雲林縣各校_英語文4年級_平均.xlsx')

data1 = data1.drop(columns='學校代碼')
#label encoding
schools_to_map1 = ["縣立饒平國小",
   "縣立大美國小","縣立東和國中","縣立雲林國中","縣立後埔國小","縣立東勢國小",
   "縣立四湖國小","縣立林厝國小","縣立鎮西國小","縣立文安國小","縣立虎尾國中",
   "縣立中正國小","縣立石榴國中","縣立石榴國小","縣立鎮南國小"]
schools_to_map2 = ['縣立大埤國小', '縣立安定國小', '縣立馬光國中', '縣立馬光國小']

data1.loc[data1['學校名稱'].isin(schools_to_map1), '學校名稱'] = "ETA"
data1.loc[data1['學校名稱'].isin(schools_to_map2), '學校名稱'] = "ETF"
data1.loc[~data1['學校名稱'].isin(["ETA","ETF"]), '學校名稱'] = "else"

data1.rename(columns={
    '整體答對率(%)': '整體答對率',
    '聽力-語音聽辨(%)': '聽力_語音聽辨',
    '聽力-辭彙聽辨(%)': '聽力_辭彙聽辨',
    '聽力-教室生活辭句理解(%)': '聽力_教室生活辭句理解',
    '聽力-文化節慶理解(%)': '聽力_文化節慶理解',
    '閱讀能力-字詞辨識(%)': '閱讀能力_字詞辨識',
    '閱讀能力-句子理解(%)': '閱讀能力_句子理解',
    '閱讀能力-文化節慶理解(%)': '閱讀能力_文化節慶理解'}, inplace=True)


manova = MANOVA.from_formula(
    '整體答對率 + 聽力_語音聽辨 + 聽力_辭彙聽辨 + 聽力_教室生活辭句理解 + 聽力_文化節慶理解 + 閱讀能力_字詞辨識 + 閱讀能力_句子理解 + 閱讀能力_文化節慶理解 ~ 學校名稱',
    data=data1)
                             
result = manova.mv_test()
#print(result)

#anova
#oversampleling/先分群
data1 = data1.drop(['整體答對率'], axis=1)
sectors = data1.groupby("學校名稱")
d1=sectors.get_group("ETF")
d2=sectors.get_group("ETA")
data_else=sectors.get_group("else")
data_etf_eta = pd.concat([d1, d2], axis=0, ignore_index=True)

data_etf_eta = data_etf_eta.drop(['學校名稱'], axis=1)
data_else = data_else.drop(['學校名稱'], axis=1)

# 根據總分劃分高分、低分和一般分組
data_etf_eta['Total'] = data_etf_eta.sum(axis=1)
data_etf_eta['label'] = pd.qcut(data_etf_eta['Total'], q=4, labels=['low', 'medium1','medium2', 'high'])

# 2. 使用 RandomOverSampler 進行重抽樣
X = data_etf_eta.drop(['label'], axis=1)
y = data_etf_eta['label']

ros = RandomOverSampler(sampling_strategy='not majority')
X_res, y_res = ros.fit_resample(X, y)

# 2. 将重采样后的数据转换为 DataFrame
resampled_data = pd.DataFrame(X_res, columns=X.columns)
resampled_data['label'] = y_res
#生成的虛擬資料
final_sample = resampled_data.sample(n=127, replace=True, random_state=42)
data_etf_eta = pd.concat([data_etf_eta, final_sample], axis=0, ignore_index=True)
data_etf_eta = data_etf_eta.drop(['Total','label'], axis=1)

#Levene's test
statistic, p_value=levene(data_etf_eta["聽力_語音聽辨"], data_else["聽力_語音聽辨"])
#print('statistic:%.2f'%statistic)
#print('p_value:%.2f'%p_value)

#welth anova
import pingouin as pg
score_lista = data_etf_eta["聽力_語音聽辨"].tolist()
score_listb = data_else["聽力_語音聽辨"].tolist()

# create DataFrame 
df = pd.DataFrame({'score': score_lista + score_listb , 
                   'group': np.repeat(['data_etf_eta', 'data_else'], 
                                      repeats=141)}) 

# perform Welch's ANOVA 
anova_result=pg.welch_anova(dv='score', between='group', data=df) 
#print(anova_result)



#Levene's test
statistic, p_value=levene(data_etf_eta["聽力_語音聽辨"], data_else["聽力_語音聽辨"])
print('statistic:%.2f'%statistic)
print('p_value:%.2f'%p_value)


#Levene's test
statistic, p_value=levene(data_etf_eta["聽力_辭彙聽辨"], data_else["聽力_辭彙聽辨"])
print("聽力_辭彙聽辨",'statistic:%.2f'%statistic)
print("聽力_辭彙聽辨",'p_value:%.2f'%p_value)



#Levene's test
statistic, p_value=levene(data_etf_eta["聽力_教室生活辭句理解"], data_else["聽力_教室生活辭句理解"])
print("聽力_教室生活辭句理解",'statistic:%.2f'%statistic)
print("聽力_教室生活辭句理解",'p_value:%.2f'%p_value)



#Levene's test
statistic, p_value=levene(data_etf_eta["聽力_文化節慶理解"], data_else["聽力_文化節慶理解"])
print("聽力_文化節慶理解",'statistic:%.2f'%statistic)
print("聽力_文化節慶理解",'p_value:%.2f'%p_value)



#Levene's test
statistic, p_value=levene(data_etf_eta["閱讀能力_字詞辨識"], data_else["閱讀能力_字詞辨識"])
print("閱讀能力_字詞辨識",'statistic:%.2f'%statistic)
print("閱讀能力_字詞辨識",'p_value:%.2f'%p_value)

#Levene's test
statistic, p_value=levene(data_etf_eta["閱讀能力_句子理解"], data_else["閱讀能力_句子理解"])
print("閱讀能力_句子理解",'statistic:%.2f'%statistic)
print("閱讀能力_句子理解",'p_value:%.2f'%p_value)


#Levene's test
statistic, p_value=levene(data_etf_eta["閱讀能力_字詞辨識"], data_else["閱讀能力_字詞辨識"])
print("閱讀能力_字詞辨識",'statistic:%.2f'%statistic)
print("閱讀能力_字詞辨識",'p_value:%.2f'%p_value)
#"閱讀能力_文化節慶理解"

#Levene's test
statistic, p_value=levene(data_etf_eta["閱讀能力_文化節慶理解"], data_else["閱讀能力_文化節慶理解"])
print("閱讀能力_文化節慶理解",'statistic:%.2f'%statistic)
print("閱讀能力_文化節慶理解",'p_value:%.2f'%p_value)

#------------------------------------------------------------------
#welth anova
import pingouin as pg
score_lista = data_etf_eta["聽力_辭彙聽辨"].tolist()
score_listb = data_else["聽力_辭彙聽辨"].tolist()

# create DataFrame 
df = pd.DataFrame({'score': score_lista + score_listb , 
                   'group': np.repeat(['data_etf_eta', 'data_else'], 
                                      repeats=141)}) 

# perform Welch's ANOVA 
anova_result=pg.welch_anova(dv='score', between='group', data=df) 
print(anova_result)

#------------------------------------------------------------------
#welth anova
import pingouin as pg
score_lista = data_etf_eta["聽力_教室生活辭句理解"].tolist()
score_listb = data_else["聽力_教室生活辭句理解"].tolist()

# create DataFrame 
df = pd.DataFrame({'score': score_lista + score_listb , 
                   'group': np.repeat(['data_etf_eta', 'data_else'], 
                                      repeats=141)}) 

# perform Welch's ANOVA 
anova_result=pg.welch_anova(dv='score', between='group', data=df) 
print(anova_result)

#------------------------------------------------------------------
#welth anova
import pingouin as pg
score_lista = data_etf_eta["聽力_文化節慶理解"].tolist()
score_listb = data_else["聽力_文化節慶理解"].tolist()

# create DataFrame 
df = pd.DataFrame({'score': score_lista + score_listb , 
                   'group': np.repeat(['data_etf_eta', 'data_else'], 
                                      repeats=141)}) 

# perform Welch's ANOVA 
anova_result=pg.welch_anova(dv='score', between='group', data=df) 
print(anova_result)

#------------------------------------------------------------------
#welth anova
import pingouin as pg
score_lista = data_etf_eta["閱讀能力_字詞辨識"].tolist()
score_listb = data_else["閱讀能力_字詞辨識"].tolist()

# create DataFrame 
df = pd.DataFrame({'score': score_lista + score_listb , 
                   'group': np.repeat(['data_etf_eta', 'data_else'], 
                                      repeats=141)}) 

# perform Welch's ANOVA 
anova_result=pg.welch_anova(dv='score', between='group', data=df) 
print(anova_result)

#------------------------------------------------------------------
#welth anova
import pingouin as pg
score_lista = data_etf_eta["閱讀能力_句子理解"].tolist()
score_listb = data_else["閱讀能力_句子理解"].tolist()

# create DataFrame 
df = pd.DataFrame({'score': score_lista + score_listb , 
                   'group': np.repeat(['data_etf_eta', 'data_else'], 
                                      repeats=141)}) 

# perform Welch's ANOVA 
anova_result=pg.welch_anova(dv='score', between='group', data=df) 
print(anova_result)

#------------------------------------------------------------------
#welth anova
import pingouin as pg
score_lista = data_etf_eta["閱讀能力_字詞辨識"].tolist()
score_listb = data_else["閱讀能力_字詞辨識"].tolist()

# create DataFrame 
df = pd.DataFrame({'score': score_lista + score_listb , 
                   'group': np.repeat(['data_etf_eta', 'data_else'], 
                                      repeats=141)}) 

# perform Welch's ANOVA 
anova_result=pg.welch_anova(dv='score', between='group', data=df) 
print(anova_result)

#------------------------------------------------------------------
#welth anova
import pingouin as pg
score_lista = data_etf_eta["閱讀能力_文化節慶理解"].tolist()
score_listb = data_else["閱讀能力_文化節慶理解"].tolist()

# create DataFrame 
df = pd.DataFrame({'score': score_lista + score_listb , 
                   'group': np.repeat(['data_etf_eta', 'data_else'], 
                                      repeats=141)}) 

# perform Welch's ANOVA 
anova_result=pg.welch_anova(dv='score', between='group', data=df) 
print(anova_result)

















