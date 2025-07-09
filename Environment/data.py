import pandas as pd
import numpy as np
from sklearn import linear_model
from scipy import stats
from fitter import Fitter, get_common_distributions
from cfg_basic import *


class DataGenerator:
    def __init__(self, num_blocks=50):
        self.cfg = Configure()
        self.num_blocks = num_blocks

        # 없는 데이터를 전부 제거한 데이터프레임 생성(이후 이 데이터프레임을 사용)
        self.df_revised_for_group = pd.read_excel(self.cfg.data_params['data_revised_filepath'])

        # 그룹 개수에 대한 데이터프레임
        self.df_group_count = pd.read_excel(self.cfg.data_params['data_group'])

        # 그룹별 중량 모델에 대한 데이터프레임
        self.df_W_model = pd.read_excel(self.cfg.data_params['model_for_W'])

        # 그룹별 H01 모델에 대한 데이터프레임
        self.df_H01_model = pd.read_excel(self.cfg.data_params['model_for_H01'])

        # 그룹별 H02 모델에 대한 데이터프레임
        self.df_H02_model = pd.read_excel(self.cfg.data_params['model_for_H02'])

        # 그룹별 duration 모델에 대한 데이터프레임
        self.df_duration_model = pd.read_excel(self.cfg.data_params['model_for_duration'])

        # 착수일 간격에 대한 데이터프레임
        self.df_intervals_count = pd.read_excel('../data/착수일 간격_피팅완료(수정).xlsx')

        self.df_buffer = pd.read_excel('../data/geometric_buffer_중조.xlsx')


    def generate_group(self):  # 그룹을 선택한 후 선종과 블록 종류로 나누기 위한 함수
        # 그룹을 랜덤으로 선택->선종과 블록 타입으로 분리
        group = np.random.choice(self.df_group_count['선종_블록'], p=self.df_group_count['Proportion'])
        ship_type = group[0:2]
        block_type = group[-1]

        return (group, ship_type, block_type)


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


    def calculate_interval(self):  # 착수일 간격 계산을 위한 함수
        p = self.df_intervals_count.loc[0, 'Proportion']
        interval = stats.geom.rvs(p) - 1  # scipy의 geometric 함수의 rvs는 1부터 시작하기 때문에 1을 빼서 사용

        return interval


    def generate_property(self, group_code, process_type, property):
        df_code = self.df_revised_for_group[self.df_revised_for_group['선종_블록'] == group_code]
        df_code_process = df_code[df_code['공종_명칭'] == process_type]
        df_for_fit = df_code_process[['선종_블록', property]]
        df_for_fit_count = pd.DataFrame(df_for_fit[property].value_counts())
        df_for_fit_count.reset_index(inplace=True)
        df_for_fit_count.sort_values(property, inplace=True)

        data = []
        for p, c in zip(df_for_fit_count[property], df_for_fit_count['count']):
            data.extend([p]*c)
        data = np.array(data)
        distributions_list = get_common_distributions()

        f = Fitter(data, distributions=distributions_list)
        f.fit()

        best_distribution = f.get_best()
        best_distribution_name = list(best_distribution.keys())[0]
        best_params = f.fitted_param[best_distribution_name]

        rvs = 0

        if best_distribution_name == 'cauchy':
            rvs = stats.cauchy.rvs(*best_params)
        elif best_distribution_name == 'expon':
            rvs = stats.expon.rvs(*best_params)
        elif best_distribution_name == 'gamma':
            rvs = stats.gamma.rvs(*best_params)
        elif best_distribution_name == 'norm':
            rvs = stats.norm.rvs(*best_params)
        elif best_distribution_name == 'exponpow':
            rvs = stats.exponpow.rvs(*best_params)
        elif best_distribution_name == 'lognorm':
            rvs = stats.lognorm.rvs(*best_params)
        elif best_distribution_name == 'powerlaw':
            rvs = stats.powerlaw.rvs(*best_params)
        elif best_distribution_name == 'reyleigh':
            rvs = stats.reyleigh.rvs(*best_params)
        elif best_distribution_name == 'uniform':
            rvs = stats.uniform.rvs(*best_params)

        property_value = rvs

        if property_value > df_for_fit_count[property].max():
            property_value = df_for_fit_count[property].max()
        elif property_value < df_for_fit_count[property].min():
            property_value = df_for_fit_count[property].min()

        property_value = np.floor(property_value * 10) / 10

        return property_value


    def generate_weight(self, group_code, process_type, length, breadth, height):
        if group_code not in ['CN_T', 'LN_D', 'VL_D']:
            df_revised_for_weight = self.df_revised_for_group[self.df_revised_for_group['선종_블록'] == group_code]

            idx_group = self.df_W_model[self.df_W_model['선종_블록'] == group_code].index

        else:
            if group_code == 'CN_T':
                df_revised_for_weight = self.df_revised_for_group[self.df_revised_for_group['선종_블록'] == 'CN_D']
                idx_group = self.df_W_model[self.df_W_model['선종_블록'] == 'CN_D'].index
            elif group_code == 'LN_D':
                df_revised_for_weight = self.df_revised_for_group[self.df_revised_for_group['선종_블록'] == 'LN_E']
                idx_group = self.df_W_model[self.df_W_model['선종_블록'] == 'LN_E'].index
            elif group_code == 'VL_D':
                df_revised_for_weight = self.df_revised_for_group[self.df_revised_for_group['선종_블록'] == 'VL_B']
                idx_group = self.df_W_model[self.df_W_model['선종_블록'] == 'VL_B'].index

        reg_coef = self.df_W_model.loc[idx_group, 'coef'].values[0]
        noise = self.df_W_model.loc[idx_group, 'std'].values[0]

        df_revised_for_final = df_revised_for_weight[df_revised_for_weight['공종_명칭'] == 'Final조립']
        max_weight = df_revised_for_final['W'].max()
        min_weight = df_revised_for_final['W'].min()

        LBH_value = length * breadth * height

        if process_type == 'Final조립':
            weight = reg_coef * LBH_value + np.random.normal(0, noise)

        # 중조 무게 피팅
        else:
            y_pred = reg_coef * LBH_value
            max_limit = y_pred * self.cfg.data_params['weight_max_limit_ratio']

            weight = max_limit + np.random.normal(0, noise)

        if weight < min_weight:
            weight = min_weight
        elif weight > max_weight:
            weight = max_weight

        weight = np.int64(weight)

        return weight


    def generate_workload_h01(self, group_code, length, breadth):
        df_for_H01 = self.df_revised_for_group[self.df_revised_for_group['선종_블록'] == group_code]
        idx_group = self.df_H01_model[self.df_H01_model['선종_블록'] == group_code].index

        if group_code == 'VL_D':
            workload_h01 = np.random.choice([189, 193], p=[0.5, 0.5])
        else:
            min_limit = df_for_H01['H01'].min()

            reg_coef = [self.df_H01_model.loc[idx_group, 'coef_0'].values[0], self.df_H01_model.loc[idx_group, 'coef_1'].values[0], self.df_H01_model.loc[idx_group, 'coef_2'].values[0]]
            noise = self.df_H01_model.loc[idx_group, 'std'].values[0]

            workload_h01 = reg_coef[0] * length + reg_coef[1] * breadth + reg_coef[2] * (length * breadth) + np.random.normal(0, noise)

            if workload_h01 < min_limit:
                workload_h01 = min_limit

            workload_h01 = np.int64(workload_h01)

        return workload_h01


    def generate_workload_h02(self, group_code, workload_h01):     # H01에 비례
        df_for_H02 = self.df_revised_for_group[self.df_revised_for_group['선종_블록'] == group_code]
        idx_group = self.df_H02_model[self.df_H02_model['선종_블록'] == group_code].index

        min_limit = df_for_H02['H02'].min()
        max_limit = df_for_H02['H02'].max()

        reg_coef = self.df_H02_model.loc[idx_group, 'coef'].values[0]
        noise = self.df_H02_model.loc[idx_group, 'std'].values[0]

        workload_h02 = reg_coef * workload_h01 + np.random.normal(0, noise)
        if workload_h02 < min_limit:
            workload_h02 = min_limit
        elif workload_h02 > max_limit:
            workload_h02 = max_limit

        workload_h02 = np.int64(workload_h02)

        return workload_h02


    def generate_duration(self, group_code, workload_H01, workload_H02, weight):
        df_for_duration = self.df_revised_for_group[self.df_revised_for_group['선종_블록'] == group_code]
        min_limit = df_for_duration['계획공기'].min()

        idx_group = self.df_duration_model[self.df_duration_model['선종_블록'] == group_code].index
        reg_coef = [self.df_duration_model.loc[idx_group, 'coef_0'].values[0],
                    self.df_duration_model.loc[idx_group, 'coef_1'].values[0],
                    self.df_duration_model.loc[idx_group, 'coef_2'].values[0]]
        noise = self.df_duration_model.loc[idx_group, 'std'].values[0]


        duration = reg_coef[0] * workload_H01 + reg_coef[1] * workload_H02  + reg_coef[2] * weight + np.random.normal(0, noise)

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


    def generate(self, file_path='../data/데이터 생성 예시_12.xlsx'):
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

            length = self.generate_property(group_code, process_type, 'L')
            breadth = self.generate_property(group_code, process_type, 'B')
            height = self.generate_property(group_code, process_type, 'H')

            weight = self.generate_weight(group_code, process_type, length, breadth, height)

            workload_h01 = self.generate_workload_h01(group_code, length, breadth)
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

data_gen = DataGenerator()
df_blocks = data_gen.generate()