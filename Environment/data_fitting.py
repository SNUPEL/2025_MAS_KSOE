import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from scipy.stats import geom
from scipy.stats import beta
from scipy.stats import lognorm


# data_group.py에서 생성한 데이터를 불러옴
df = pd.read_excel('../data/블록-계획데이터(예제)_블록 그룹 추가.xlsx')

# 2. 착수계획에 대한 착수일 간격 계산
df_startplan = df['착수계획']

# 착수일 개수 계산
df_startplan_count = pd.DataFrame(df_startplan.value_counts())
df_startplan_count.sort_index(inplace = True)
df_startplan_count.reset_index(inplace = True)

# 날짜를 정수로 환산
date_range = pd.date_range(df_startplan_count.loc[0, '착수계획'], df_startplan_count.loc[759, '착수계획'])
date_range = list(date_range)
changed_date = []

for i in range(len(date_range)):
    for j in range(df_startplan_count.shape[0]):
        if df_startplan_count.loc[j, '착수계획'] == date_range[i]:
            changed_date.append(i)

df_startplan_count.insert(loc=1, column='changed_date', value=changed_date)
df_startplan_count.to_excel('../data/착수일 날짜 변환.xlsx')

# 간격 계산
intervals = []

for i in range(df_startplan_count.shape[0]):
    if i == 0:
        continue
    else:
        interval = df_startplan_count.loc[i, 'changed_date'] - df_startplan_count.loc[i - 1, 'changed_date']
        intervals.append(interval)

df_intervals = pd.DataFrame({'Interval': intervals})

# 간격이 나오는 횟수 계산
df_intervals_count = pd.DataFrame(df_intervals.value_counts())
df_intervals_count.sort_index(inplace = True)
df_intervals_count.reset_index(inplace = True)

# 비율 계산
df_intervals_count['Proportion'] = df_intervals_count['count'] / df_intervals_count['count'].sum()

# 비율에 대한 geometric fitting
pmf_geom = geom.pmf(df_intervals_count['Interval'], p=df_intervals_count.loc[0, 'Proportion'])
df_intervals_count['fit_result'] = pmf_geom

df_intervals_count.to_excel('../data/착수일 간격_geometric 분포 적합.xlsx')


# 3. 블록 제원(L, B, H) 피팅
# 선종_블록 코드별로 원하는 제원에 대하여 피팅을 하기 위한 함수, beta 피팅과 lognorm 피팅에 대응 가능
def making_model_property(property, code, fitting_method):
    df_L = pd.DataFrame(df[['선종_블록', 'L']])
    df_B = pd.DataFrame(df[['선종_블록', 'B']])
    df_H = pd.DataFrame(df[['선종_블록', 'H']])

    if property == 'L':
        df_code = df_L[df_L['선종_블록'] == code]

    elif property == 'B':
        df_code = df_B[df_B['선종_블록'] == code]

    elif property == 'H':
        df_code = df_H[df_H['선종_블록'] == code]

    df_code_count = pd.DataFrame(df_code.value_counts())
    df_code_count.reset_index(inplace=True)
    df_code_count.sort_values(property, inplace=True)
    df_code_count['Density'] = df_code_count['count'] / df_code_count['count'].sum()


    if fitting_method == 'beta':
        df_forbeta = df_code_count[[property, 'Density']]
        array_forbeta = df_forbeta.to_numpy()
        a, b, loc, scale = beta.fit(array_forbeta)
        pdf_beta = beta.pdf(df_code_count[property], a, b, loc, scale)
        df_code_count['fit_result'] = pdf_beta

    elif fitting_method == 'lognorm':
        df_lognorm = df_code_count[[property, 'Density']]
        array_lognorm = df_lognorm.to_numpy()
        s, loc, scale = lognorm.fit(array_lognorm[:, 1])
        pdf_lognorm = lognorm.pdf(df_code_count[property], s, loc, scale)
        df_code_count['fit_result'] = pdf_lognorm

    return df_code_count

df_group_count = pd.DataFrame(df['선종_블록'].value_counts())



# beta 피팅 사용
with pd.ExcelWriter('../data/블록 제원 피팅_L.xlsx') as writer:
    for code in df_group_count.index:
        df_property_code = making_model_property('L', code, 'beta')
        df_property_code.to_excel(writer, sheet_name=code)

with pd.ExcelWriter('../data/블록 제원 피팅_B.xlsx') as writer:
    for code in df_group_count.index:
        df_property_code = making_model_property('B', code, 'beta')
        df_property_code.to_excel(writer, sheet_name=code)

with pd.ExcelWriter('../data/블록 제원 피팅_H.xlsx') as writer:
    for code in df_group_count.index:
        df_property_code = making_model_property('H', code, 'beta')
        df_property_code.to_excel(writer, sheet_name=code)


# 4. W 피팅
# 선종_블록 코드별로 W 피팅을 하기 위한 함수. a(L * B * H) + b 형식
def making_model_weight(code):
    df_LBHW = df[['선종_블록', 'W', 'L', 'B', 'H']]
    df_LBHW['LBH'] = df_LBHW['L'] * df_LBHW['B'] * df_LBHW['H']
    df_weight = df_LBHW[df_LBHW['선종_블록'] == code]
    x = df_weight['LBH']
    x = x.to_numpy()
    x = x.reshape(-1, 1)
    y = df_weight['W']
    y = y.to_numpy()

    reg = linear_model.LinearRegression()
    reg.fit(x, y)

    df_weight['fit_result'] = reg.coef_[0] * df_weight['LBH'] + reg.intercept_
    return df_weight

with pd.ExcelWriter('../data/중량 피팅.xlsx') as writer:
    for code in df_group_count.index:
        df_weight_code = making_model_weight(code)
        df_weight_code.to_excel(writer, sheet_name=code)


# 5. H01, H02 피팅
# 선종_블록 코드별로 H01, H02 피팅을 하기 위한 함수, (H01 or H02) = aW + bL + cB + dH
def making_model_working(code):
    df_work = df[['선종_블록', 'H01', 'H02', 'W', 'L', 'B', 'H']]
    df_work_code = df_work[df_work['선종_블록'] == code]

    x = df_work_code[['W', 'L', 'B', 'H']]
    y_1 = df_work_code['H01']
    y_2 = df_work_code['H02']

    reg_1 = linear_model.LinearRegression()
    reg_2 = linear_model.LinearRegression()
    reg_1.fit(x, y_1)
    reg_2.fit(x, y_2)

    df_work_code['fit_result_H01'] = reg_1.coef_[0] * df_work_code['W'] + reg_1.coef_[1] * df_work_code['L'] + reg_1.coef_[2] * df_work_code['B'] + reg_1.coef_[3] * df_work_code['H'] + reg_1.intercept_
    df_work_code['fit_result_H02'] = reg_2.coef_[0] * df_work_code['W'] + reg_2.coef_[1] * df_work_code['L'] + reg_2.coef_[2] * df_work_code['B'] + reg_2.coef_[3] * df_work_code['H'] + reg_2.intercept_

    return df_work_code

with pd.ExcelWriter('../data/작업량 피팅.xlsx') as writer:
    for code in df_group_count.index:
        df_work_code = making_model_working(code)
        df_work_code.to_excel(writer, sheet_name=code)


# 6. 작업공기 피팅
# 선종_블록 코드별로 작업공기 피팅을 하기 위한 함수. (계획공기) = a * H01 + b * H02
# 계산된 작업공기는 반올림하여 정수형으로 사용
def making_model_duration_plan(code):
    df_duration_plan = df[['선종_블록', '계획공기', 'H01', 'H02']]
    df_group_duration = df_duration_plan[df_duration_plan['선종_블록'] == code]
    x = df_group_duration[['H01', 'H02']]
    y = df_group_duration['계획공기']

    reg = linear_model.LinearRegression()
    reg.fit(x, y)

    df_group_duration['fit_result'] = reg.coef_[0] * df_group_duration['H01'] + reg.coef_[1] * df_group_duration['H02'] + reg.intercept_
    df_group_duration['fit_result'] = round(df_group_duration['fit_result'], 0)

    return df_group_duration

with pd.ExcelWriter('../data/작업공기 피팅.xlsx') as writer:
    for code in df_group_count.index:
        df_duration_code = making_model_duration_plan(code)
        df_duration_code.to_excel(writer, sheet_name=code)