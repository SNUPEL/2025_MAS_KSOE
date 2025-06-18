import numpy as np
import pandas as pd


df = pd.read_excel('../data/블록-계획데이터(예제)_수정.xlsx')      # 전체 데이터에 대한 데이터프레임


# 1. 선종_블록 그룹 생성
group_list = []
for i in range(12132):      # 전체 데이터 12132개에 대하여 탐색 후 그룹 생성
    group_code = df.loc[i, '선종_코드'] + '_' + df.loc[i, '블록'][0]
    group_list.append(group_code)
df_addgroup = df        # 그룹을 더할 데이터프레임 새로 정의
df_addgroup['선종_블록'] = group_list
df_addgroup.to_excel('../data/블록-계획데이터(예제)_블록 그룹 추가.xlsx')      # 선종_블록이 추가된 형태 엑셀파일로 저장

# 선종_블록 그룹 비율 계산
df_group_count = pd.DataFrame(df_addgroup['선종_블록'].value_counts())
df_group_count['proportion'] = df_group_count['count'] / df_group_count['count'].sum()      # 비율 계산
df_group_count.to_excel('../data/선종_블록 그룹 간 비율.xlsx')       # 블록 별 비율 계산 결과 엑셀 데이터 저장

# 실행할 때는 data_group.py를 먼저 실행시킨 후 data_fitting.py를 실행