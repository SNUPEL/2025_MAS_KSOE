import pandas as pd

# 취부/용접 작업에 맞춰서 취부/용접팀 이름과 정반_코드에 맞도록 데이터프레임을 생성하는 함수
# 정반_코드가 없으면 팀 자체의 데이터프레임을 뽑음
def generating_df_team_name(work, team_name, factory=None):
    df = pd.read_excel('../data/블록-계획데이터(예제)_수정.xlsx')

    df['착수실적'] = pd.to_datetime(df['착수실적'], errors='coerce')
    df['완료실적'] = pd.to_datetime(df['완료실적'], errors='coerce')
    if work == 'H01':
        df_per_day_1 = df[['정반_코드', '취부팀_코드', '착수실적', '완료실적', '실적공기', 'H01']]
        if factory:
            df_per_day_1 = df_per_day_1[df_per_day_1['정반_코드'] == factory]
        else:
            df_per_day_1 = df[['취부팀_코드', '착수실적', '완료실적', '실적공기', 'H01']]

        idx_1 = df_per_day_1[df_per_day_1['취부팀_코드'] == 'XXX'].index
        df_per_day_1 = df_per_day_1.drop(idx_1)
        df_per_day_1 = df_per_day_1.dropna(axis=0)

        df_per_day_1['하루_취부'] = df_per_day_1['H01'] / df_per_day_1['실적공기']

        df_per_day_1_ex = df_per_day_1[df_per_day_1['취부팀_코드'] == team_name]
        df_team_name = df_per_day_1_ex

    elif work == 'H02':
        df_per_day_2 = df[['정반_코드', '용접팀_코드', '착수실적', '완료실적', '실적공기', 'H02']]
        if factory:
            df_per_day_2 = df_per_day_2[df_per_day_2['정반_코드'] == factory]

        idx_2 = df_per_day_2[df_per_day_2['용접팀_코드'] == 'XXX'].index
        df_per_day_2 = df_per_day_2.drop(idx_2)
        df_per_day_2 = df_per_day_2.dropna(axis=0)

        df_per_day_2['하루_용접'] = df_per_day_2['H02'] / df_per_day_2['실적공기']

        df_per_day_2_ex = df_per_day_2[df_per_day_2['용접팀_코드'] == team_name]
        df_team_name = df_per_day_2_ex

    return df_team_name


# 위에서 생성한 팀별 데이터프레임을 이용하여 capacity를 계산하는 함수
def make_capacity(work, team_name, factory=None):
    df = generating_df_team_name(work, team_name, factory)
    df.index = range(len(df))

    df_capacity = pd.read_excel('../data/capacity data.xlsx', sheet_name=team_name)     # 블록별 작업일자와 근무일/휴일 여부를 모은 파일. 생성 방법은 추후 생각 예정
    ser_date = pd.Series(pd.date_range(str(df_capacity.loc[0, '날짜']), str(df_capacity.loc[df_capacity.shape[0] - 1, '날짜'])))
    df_capacity['정반_코드'] = df['정반_코드']
    df_capacity['날짜'] = ser_date

    # 누적 노동량 컬럼 초기화
    df_capacity['Capacity'] = 0.0

    # 날짜 순회
    for i in range(df_capacity.shape[0]):
        today = df_capacity.loc[i, '날짜'].date()

        # 이전 날짜의 Capacity 가져오기
        if i > 0:
            df_capacity.loc[i, 'Capacity'] = df_capacity.loc[i - 1, 'Capacity']

        # 각 작업 순회
        for j in range(df.shape[0]):
            start_date = df.loc[j, '착수실적'].date()
            end_date = df.loc[j, '완료실적'].date()
            if work == 'H01':
                daily_amount = df.loc[j, '하루_취부']
            elif work == 'H02':
                daily_amount = df.loc[j, '하루_용접']

            # 착수일이면 증가
            if today == start_date:
                df_capacity.loc[i, 'Capacity'] += daily_amount

            # 완료일이면 감소 시점 결정
            if today == end_date:
                work_type = df_capacity.loc[i, '근무일 여부']

                if work_type == '근무일':
                    df_capacity.loc[i, 'Capacity'] -= daily_amount
                elif work_type == '휴일':
                    # 다음 근무일 찾아서 차감
                    for k in range(i + 1, df_capacity.shape[0]):
                        if df_capacity.loc[k, '근무일 여부'] == '근무일':
                            df_capacity.loc[k, 'Capacity'] -= daily_amount
                            break  # 한 번만 차감해야 하니까 break

    return df_capacity