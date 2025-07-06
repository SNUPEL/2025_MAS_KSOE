import pandas as pd
import numpy as np
from sklearn import linear_model
from scipy import stats
from fitter import Fitter, get_common_distributions

class DataGenerator:
    def __init__(self,
                 num_blocks=50):

        self.num_blocks = num_blocks
        self.df = pd.read_excel('../data/블록-계획데이터(예제)_수정.xlsx')

        # 없는 데이터를 전부 제거한 데이터프레임 생성(이후 이 데이터프레임을 사용)
        self.df_revised = self.df.dropna(axis=0)
        idx_1 = self.df_revised[self.df_revised['취부팀_코드'] == 'XXX'].index
        self.df_revised = self.df_revised.drop(idx_1)
        idx_2 = self.df_revised[self.df_revised['용접팀_코드'] == 'XXX'].index
        self.df_revised = self.df_revised.drop(idx_2)
        idx_3 = self.df_revised[self.df_revised['stage_코드'] == 'XXX'].index
        self.df_revised = self.df_revised.drop(idx_3)
        idx_4 = self.df_revised[self.df_revised['정반_코드'] == 'XXXX'].index
        self.df_revised = self.df_revised.drop(idx_4)
        idx_5 = self.df_revised[self.df_revised['실적공기'] < 0].index
        self.df_revised = self.df_revised.drop(idx_5)
        idx_6 = self.df_revised[self.df_revised['L'] == 815].index
        self.df_revised = self.df_revised.drop(idx_6)
        self.df_revised.reset_index(inplace=True)

        # 공정 개수에 대한 데이터프레임
        # self.df_process_count = pd.DataFrame(self.df_revised['공종_명칭'].value_counts())
        # self.df_process_count.reset_index(inplace=True)
        # self.df_process_count['Proportion'] = self.df_process_count['count'] / self.df_process_count['count'].sum()

        # 그룹 개수에 대한 데이터프레임
        self.df_revised_for_group = self.df_revised.copy()
        group_list = []
        for i in range(self.df_revised.shape[0]):
            group_code = self.df_revised.loc[i, '선종_코드'] + '_' + self.df_revised.loc[i, '블록'][0]
            group_list.append(group_code)
        self.df_revised_for_group['선종_블록'] = group_list
        self.df_group_count = pd.DataFrame(self.df_revised_for_group['선종_블록'].value_counts())
        self.df_group_count.reset_index(inplace=True)
        self.df_group_count['Proportion'] = self.df_group_count['count'] / self.df_group_count['count'].sum()
        self.df_group_count['panel_count'] = 0
        self.df_group_count['curve_count'] = 0
        self.df_group_count['big_count'] = 0
        self.df_group_count['final_count'] = 0

        for code in self.df_group_count['선종_블록']:
            panel = 0
            curve = 0
            big = 0
            final = 0

            df_for_group = self.df_revised_for_group[self.df_revised_for_group['선종_블록'] == code]
            for process in df_for_group['공종_명칭']:
                if process == '평중조':
                    panel += 1
                elif process == '곡중조':
                    curve += 1
                elif process == '대조중조':
                    big += 1
                elif process == 'Final조립':
                    final += 1

            self.df_group_count.loc[self.df_group_count['선종_블록'] == code, 'panel_count'] = panel
            self.df_group_count.loc[self.df_group_count['선종_블록'] == code, 'curve_count'] = curve
            self.df_group_count.loc[self.df_group_count['선종_블록'] == code, 'big_count'] = big
            self.df_group_count.loc[self.df_group_count['선종_블록'] == code, 'final_count'] = final

        # 착수계획 개수에 대한 데이터프레임
        self.df_startplan_count = pd.DataFrame(self.df_revised['착수계획'].value_counts())
        self.df_startplan_count.sort_index(inplace=True)
        self.df_startplan_count.reset_index(inplace=True)
        date_range = pd.date_range(self.df_startplan_count.loc[0, '착수계획'],
                                   self.df_startplan_count.loc[self.df_startplan_count.shape[0] - 1, '착수계획'])
        date_range = list(date_range)
        changed_date = []

        for i in range(len(date_range)):
            for j in range(self.df_startplan_count.shape[0]):
                if self.df_startplan_count.loc[j, '착수계획'] == date_range[i]:
                    changed_date.append(i)

        self.df_startplan_count.insert(loc=1, column='changed_date', value=changed_date)
        self.df_startplan_count['Proportion'] = self.df_startplan_count['count'] / self.df_startplan_count[
            'count'].sum()

        # 착수일 간격에 대한 데이터프레임
        self.df_intervals_count = pd.read_excel('../data/착수일/착수일 간격_피팅완료(수정).xlsx')

        self.df_buffer = pd.read_excel('../data/geometric_buffer_중조.xlsx')


    def generate_process(self, group_code):         # 공종 명칭 생성 함수, 공정이 나오는 비율에 맞춰서 데이터 생성
        df_process_group = self.df_group_count[self.df_group_count['선종_블록'] == group_code]
        # 각 count 값을 올바르게 추출
        count = df_process_group['count'].values[0]
        panel_proportion = df_process_group['panel_count'].values[0] / count
        curve_proportion = df_process_group['curve_count'].values[0] / count
        big_proportion = df_process_group['big_count'].values[0] / count
        final_proportion = df_process_group['final_count'].values[0] / count

        proportion_list = [panel_proportion, curve_proportion, big_proportion, final_proportion]

        process_type = np.random.choice(['평중조', '곡중조', '대조중조', 'Final조립'], p=proportion_list)
        return process_type


    def generate_group(self):  # 그룹을 선택한 후 선종과 블록 종류로 나누기 위한 함수
        # 그룹화를 위한 새로운 데이터프레임 생성
        df_revised_for_group = self.df_revised
        group_list = []
        for i in range(self.df_revised.shape[0]):
            group_code = self.df_revised.loc[i, '선종_코드'] + '_' + self.df_revised.loc[i, '블록'][0]
            group_list.append(group_code)
        df_revised_for_group['선종_블록'] = group_list
        df_group_count = pd.DataFrame(df_revised_for_group['선종_블록'].value_counts())
        df_group_count.reset_index(inplace=True)
        df_group_count['Proportion'] = df_group_count['count'] / df_group_count['count'].sum()

        # 그룹을 랜덤으로 선택->선종과 블록 타입으로 분리
        group = np.random.choice(df_group_count['선종_블록'], p=df_group_count['Proportion'])
        ship_type = group[0:2]
        block_type = group[-1]

        return (group, ship_type, block_type)


    def calculate_interval(self):  # 착수일 간격 계산을 위한 함수
        p = self.df_intervals_count.loc[0, 'Proportion']
        interval = stats.geom.rvs(p) - 1  # scipy의 geometric 함수의 rvs는 1부터 시작하기 때문에 1을 빼서 사용

        return interval


    def generate_property(self, group_code, property):
        df_code = self.df_revised_for_group[self.df_revised_for_group['선종_블록'] == group_code]
        df_for_fit = df_code[['선종_블록', property]]
        df_for_fit_count = pd.DataFrame(df_for_fit[property].value_counts())
        df_for_fit_count.reset_index(inplace=True)
        df_for_fit_count.sort_values(property, inplace=True)

        df_for_fit_count['Density'] = df_for_fit_count['count'] / df_for_fit_count['count'].sum()

        data = []
        for p, c in zip(df_for_fit_count[property], df_for_fit_count['count']):
            data.extend([p] * c)
        data = np.array(data)
        distributions_list = get_common_distributions()

        f = Fitter(data, distributions=distributions_list)
        f.fit()

        best_distribution = f.get_best()
        best_distribution_name = list(best_distribution.keys())[0]
        best_params = f.fitted_param[best_distribution_name]

        property_value = 0

        if best_distribution_name == 'cauchy':
            property_value = stats.cauchy.rvs(*best_params)
        elif best_distribution_name == 'expon':
            property_value = stats.expon.rvs(*best_params)
        elif best_distribution_name == 'gamma':
            property_value = stats.gamma.rvs(*best_params)
        elif best_distribution_name == 'norm':
            property_value = stats.norm.rvs(*best_params)
        elif best_distribution_name == 'exponpow':
            property_value = stats.exponpow.rvs(*best_params)
        elif best_distribution_name == 'lognorm':
            property_value = stats.lognorm.rvs(*best_params)
        elif best_distribution_name == 'powerlaw':
            property_value = stats.powerlaw.rvs(*best_params)
        elif best_distribution_name == 'reyleigh':
            property_value = stats.reyleigh.rvs(*best_params)
        elif best_distribution_name == 'uniform':
            property_value = stats.uniform.rvs(*best_params)

        if property_value > df_for_fit_count[property].max():
            property_value = df_for_fit_count[property].max()
        elif property_value < df_for_fit_count[property].min():
            property_value = df_for_fit_count[property].min()

        property_value = np.floor(property_value * 10) / 10

        return property_value


    def generate_weight(self, group_code, process_type, length, breadth, height, smoothing=False):
        df_revised_for_weight = self.df_revised_for_group[self.df_revised_for_group['선종_블록'] == group_code]
        df_revised_for_weight['LBH'] = df_revised_for_weight['L'] * df_revised_for_weight['H'] * df_revised_for_weight['B']

        # Final조립의 선형 피팅 모델 구현(group_code에 따라 달라짐)
        df_revised_for_final = df_revised_for_weight[df_revised_for_weight['공종_명칭'] == 'Final조립']

        if smoothing:       # 데이터 스무딩 여부 결정
            smoothed = []
            half_window = 40 // 2
            data = df_revised_for_final['W']
            for i in range(len(data)):
                window_start = max(0, i - half_window)
                window_end = min(len(data), i + half_window + 1)
                window = data[window_start:window_end]
                median = np.median(window)
                smoothed.append(median)

            df_revised_for_final['smoothed_W'] = smoothed

            x = df_revised_for_final['LBH'].to_numpy()
            x = x.reshape(-1, 1)

            y = df_revised_for_final['smoothed_W'].to_numpy()

        else:
            x = df_revised_for_final['LBH'].to_numpy()
            x = x.reshape(-1, 1)
            y = df_revised_for_final['W'].to_numpy()

        reg = linear_model.LinearRegression()
        reg.fit(x, y)

        LBH_value = length * breadth * height

        if process_type == 'Final조립':
            weight = reg.coef_[0] * LBH_value + reg.intercept_ + 10 * np.random.normal()

        # 중조 무게 피팅
        else:
            if group_code in ['CN_T', 'LN_D', 'VL_D']:      # 중조만 존재하는 그룹들이기 때문에 피팅 방법 없음
                weight = 0
            else:
                y_pred = reg.coef_[0] * LBH_value + reg.intercept_
                max_limit = y_pred * 0.7
                min_limit = y_pred * 0.3
                weight = stats.expon.rvs() * max_limit + np.random.normal()
                if weight < min_limit:
                    weight = min_limit
                elif weight > max_limit:
                    weight = max_limit

        weight = np.int64(weight)

        return weight


    def generate_workload_h01(self, group_code, length, breadth, height, weight):
        df_for_H01 = self.df_revised_for_group[self.df_revised_for_group['선종_블록'] == group_code]
        df_for_H01 = df_for_H01[df_for_H01['공종_명칭'] == 'Final조립']

        min_limit = df_for_H01['H01'].min()

        x = df_for_H01[['L', 'B', 'H', 'W']]
        y = df_for_H01['H01']

        reg = linear_model.LinearRegression()
        reg.fit(x, y)

        workload_h01 = reg.coef_[0] * length + reg.coef_[1] * breadth + reg.coef_[2] * height + reg.coef_[
            3] * weight + reg.intercept_ + 10 * np.random.normal()

        if workload_h01 < min_limit:
            workload_h01 = min_limit

        workload_h01 = np.int64(workload_h01)

        return workload_h01


    def generate_workload_h02(self, group_code, workload_h01):
        df_for_H02 = self.df_revised_for_group[self.df_revised_for_group['선종_블록'] == group_code]

        min_limit = df_for_H02['H02'].min()

        x = df_for_H02['H01'].to_numpy()
        x = x.reshape(-1, 1)

        y = df_for_H02['H02']

        reg = linear_model.LinearRegression()
        reg.fit(x, y)

        workload_h02 = reg.coef_[0] * workload_h01 + reg.intercept_ + 10 * np.random.normal()
        if workload_h02 < min_limit:
            workload_h02 = min_limit

        workload_h02 = np.int64(workload_h02)

        return workload_h02


    def generate_duration(self, group_code, workload_H01, workload_H02, weight):
        df_for_duration = self.df_revised_for_group[self.df_revised_for_group['선종_블록'] == group_code]
        min_limit = df_for_duration['계획공기'].min()

        x = df_for_duration[['H01', 'H02', 'W']]
        y = df_for_duration['계획공기']

        reg = linear_model.LinearRegression()
        reg.fit(x, y)

        duration = reg.coef_[0] * workload_H01 + reg.coef_[1] * workload_H02  + reg.intercept_ + reg.coef_[2] * weight + 10 * np.random.normal()

        if duration < min_limit:
            duration = min_limit

        duration = np.int64(duration)

        return duration

    def calculate_buffer(self, process_type):  # column에 들어가는 값은 아님
        if process_type == 'Final조립':
            buffer = 2
        else:
            p = self.df_buffer.loc[0, 'Proportion']
            buffer = stats.geom.rvs(p, loc=-1)

        return buffer


    def generate(self, file_path='../data/데이터 생성 예시.xlsx'):
        columns = ["Block_Name", "Block_ID", "Process_Type", "Ship_Type", "Block_Type", "Start_Date", "Duration", "Due_Date",
                   "Workload_H01", "Workload_H02", "Weight", "Length", "Breadth", "Height"]


        df_blocks = []



        for j in range(self.num_blocks):
            name = "J-%d" % j
            id = j

            # 데이터 생성 코드 추가

            group_results = self.generate_group()          # column에 포함되지는 않음
            group_code = group_results[0]           # 그룹을 참조하는 데이터를 위한 입력변수로 사용
            ship_type = group_results[1]
            block_type = group_results[2]

            process_type = self.generate_process(group_code)

            if j == 0:
                start_date = 0  # 첫번째 착수일은 0으로 고정
            else:
                interval = self.calculate_interval()
                start_date = df_blocks[j - 1][5] + interval  # 이전 착수일에 interval을 더하는 형식으로 계산

            buffer = self.calculate_buffer(process_type)
            length = self.generate_property(group_code, 'L')
            breadth = self.generate_property(group_code, 'B')
            height = self.generate_property(group_code, 'H')
            weight = self.generate_weight(group_code, process_type, length, breadth, height)
            workload_h01 = self.generate_workload_h01(group_code, length, breadth, height, weight)
            workload_h02 = self.generate_workload_h02(group_code, workload_h01)
            duration = self.generate_duration(group_code, workload_h01, workload_h02, weight)

            due_date = start_date + duration + buffer - 1

            row = [name, id, process_type, ship_type, block_type, start_date, duration, due_date,
                   workload_h01, workload_h02, weight, length, breadth, height]

            df_blocks.append(row)

        df_blocks = pd.DataFrame(df_blocks, columns=columns)

        if file_path is not None:
            writer = pd.ExcelWriter(file_path)
            df_blocks.to_excel(writer, sheet_name="blocks", index=False)
            writer.close()

        return df_blocks
