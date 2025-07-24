import ast
import numpy as np
import pandas as pd
import scipy.stats as stats


class DataGenerator:
    def __init__(self,
                 block_data_path,
                 bay_data_path,
                 num_blocks=50,
                 time_horizon=30,
                 iat_avg=0.1,
                 buffer_avg=1.5,
                 weight_factor=0.7,
                 fix_time_horizon=False):

        self.block_data_path = block_data_path
        self.bay_data_path = bay_data_path
        self.num_blocks = num_blocks
        self.time_horizon = time_horizon
        self.iat_avg = iat_avg
        self.buffer_avg = buffer_avg
        self.weight_factor = weight_factor
        self.fix_time_horizon = fix_time_horizon

        self.df_bay = pd.read_excel(bay_data_path, sheet_name="bays", engine="openpyxl")
        self.df_count = pd.read_excel(block_data_path, sheet_name="count", engine="openpyxl") # 그룹 개수에 대한 데이터프레임
        self.df_length = pd.read_excel(block_data_path, sheet_name="length", engine="openpyxl")  # 그룹 별 블록 길이 분포에 대한 데이터프레임
        self.df_breadth = pd.read_excel(block_data_path, sheet_name="breadth", engine="openpyxl")  # 그룹 별 블록 폭 분포에 대한 데이터프레임
        self.df_height = pd.read_excel(block_data_path, sheet_name="height", engine="openpyxl")  # 그룹 별 블록 높이 분포에 대한 데이터프레임
        self.df_weight =  pd.read_excel(block_data_path, sheet_name="weight", engine="openpyxl") # 그룹 별 중량 모델에 대한 데이터프레임
        self.df_h1 = pd.read_excel(block_data_path, sheet_name="h1", engine="openpyxl") # 그룹 별 H01 모델에 대한 데이터프레임
        self.df_h2 = pd.read_excel(block_data_path, sheet_name="h2", engine="openpyxl") # 그룹 별 H02 모델에 대한 데이터프레임
        self.df_duration = pd.read_excel(block_data_path, sheet_name="duration", engine="openpyxl") # 그룹 별 duration 모델에 대한 데이터프레임
        self.df_sample = pd.read_excel(block_data_path, sheet_name="sample", engine="openpyxl")

    def generate_group(self):  # 그룹을 선택한 후 선종과 블록 종류로 나누기 위한 함수
        # 그룹을 랜덤으로 선택->선종과 블록 타입으로 분리
        group_code = np.random.choice(self.df_count['group'], p=self.df_count['proportion'])
        ship_type = group_code[0:2]
        block_type = group_code[-1]

        return group_code, ship_type, block_type

    def generate_process(self, group_code):         # 공종 명칭 생성 함수, 공정이 나오는 비율에 맞춰서 데이터 생성
        df_process_count = self.df_count[self.df_count['group'] == group_code]

        # 각 count 값을 올바르게 추출
        count = df_process_count['count'].values[0]
        panel_proportion = df_process_count['panel_count'].values[0] / count
        curve_proportion = df_process_count['curve_count'].values[0] / count
        big_proportion = df_process_count['big_count'].values[0] / count
        final_proportion = df_process_count['final_count'].values[0] / count

        proportion_list = [panel_proportion, curve_proportion, big_proportion, final_proportion]
        process_type = np.random.choice(['평중조', '곡중조', '대조중조', 'Final조립'], p=proportion_list)

        return process_type

    def generate_property(self, group_code, process_type, property='L'):
        if property == 'L':
            df_property = self.df_length.copy(deep=False)
        elif property == 'B':
            df_property = self.df_breadth.copy(deep=False)
        elif property == 'H':
            df_property = self.df_height.copy(deep=False)
        else:
            raise Exception("Invalid property")

        df_property['best_params'] = df_property['best_params'].apply(ast.literal_eval)
        df_property = df_property[(df_property['group'] == group_code)
                                  & (df_property['process_type'] == process_type)]
        best_distribution_name = df_property['best_distribution_name'].values[0]
        best_params = df_property['best_params'].values[0]
        min_value = df_property['min'].values[0]
        max_value = df_property['max'].values[0]

        if best_distribution_name == 'cauchy':
            property_value = stats.cauchy.rvs(*best_params)
        elif best_distribution_name == 'chi2':
            property_value = stats.chi2.rvs(*best_params)
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
        else:
            property_value = 0
            # raise Exception("Invalid distriution")

        if property_value > max_value:
            property_value = max_value
        elif property_value < min_value:
            property_value = min_value

        property_value = np.floor(property_value * 10) / 10

        return property_value

    def generate_weight(self, group_code, process_type, length, breadth, height):
        if group_code not in ['CN_T', 'LN_D', 'VL_D']:
            df_weight = self.df_weight[self.df_weight['group'] == group_code]
        else:
            if group_code == 'CN_T': # CN_T: CN_D의 모델 사용
                df_weight = self.df_weight[self.df_weight['group'] == 'CN_D']
            elif group_code == 'LN_D': # LN_D: LN_E의 모델 사용
                df_weight = self.df_weight[self.df_weight['group'] == 'LN_E']
            elif group_code == 'VL_D': # VL_D: VL_B의 모델 사용
                df_weight = self.df_weight[self.df_weight['group'] == 'VL_B']

        reg_coef = df_weight['coef'].values[0]
        noise = df_weight['std'].values[0]
        min_value = df_weight['min'].values[0]
        max_value = df_weight['max'].values[0]

        volume = length * breadth * height

        if process_type == 'Final조립':
            weight = reg_coef * volume + np.random.normal(0, noise)
        else: # 중조 무게 피팅
            weight = reg_coef * volume * self.weight_factor + np.random.normal(0, noise)

        if weight < min_value:
            weight = min_value
        elif weight > max_value:
            weight = max_value

        weight = np.int64(weight)

        return weight

    def generate_workload_h1(self, group_code, length, breadth):
        df_h1 = self.df_h1[self.df_h1['group'] == group_code]

        reg_coef = [df_h1['coef_0'].values[0],
                    df_h1['coef_1'].values[0],
                    df_h1['coef_2'].values[0]]
        noise = df_h1['std'].values[0]
        min_value = df_h1['min'].values[0]

        workload_h1 = (reg_coef[0] * length + reg_coef[1] * breadth
                       + reg_coef[2] * (length * breadth) + np.random.normal(0, noise))

        if workload_h1 < min_value:
            workload_h1 = min_value

        workload_h1 = np.int64(workload_h1)

        return workload_h1

    def generate_workload_h2(self, group_code, workload_h1):
        df_h2 = self.df_h2[self.df_h2['group'] == group_code]

        reg_coef = df_h2['coef'].values[0]
        noise = df_h2['std'].values[0]
        min_value = df_h2['min'].values[0]
        max_value = df_h2['max'].values[0]

        workload_h2 = reg_coef * workload_h1 + np.random.normal(0, noise)

        if workload_h2 < min_value:
            workload_h2 = min_value
        elif workload_h2 > max_value:
            workload_h2 = max_value

        workload_h2 = np.int64(workload_h2)

        return workload_h2

    def generate_duration(self, group_code, workload_h1, workload_h2, weight):
        df_duration = self.df_duration[self.df_duration['group'] == group_code]

        reg_coef = [df_duration['coef_0'].values[0],
                    df_duration['coef_1'].values[0],
                    df_duration['coef_2'].values[0]]
        noise = df_duration['std'].values[0]
        min_value = df_duration['min'].values[0]

        duration = (reg_coef[0] * workload_h1 + reg_coef[1] * workload_h2
                    + reg_coef[2] * weight + np.random.normal(0, noise))

        if duration < min_value:
            duration = min_value

        duration = np.int64(duration)

        return duration

    def calculate_buffer(self, process_type):  # column에 들어가는 값은 아님
        if process_type == 'Final조립':
            buffer = 2
        else:
            p = 1 / (1 + self.buffer_avg)
            buffer = stats.geom.rvs(p) - 1

        return buffer

    def check_eligibility(self, group_code, process_type, breadth, height, weight):
        df_eligible_bay = self.df_bay[(breadth <= self.df_bay["block_breadth"]) &
                                      (height <= self.df_bay["block_height"]) &
                                      (weight <= self.df_bay["block_weight"])]

        df_possible_bay = self.df_breadth.copy()
        df_possible_bay['bay'] = df_possible_bay['bay'].apply(ast.literal_eval)

        df_possible_bay = df_possible_bay[df_possible_bay['group'] == group_code]
        df_possible_bay = df_possible_bay[df_possible_bay['process_type'] == process_type]

        if len(df_eligible_bay) > 0:
            bay_name_series = df_eligible_bay['bay_name']


        elif len(df_eligible_bay) == 0:

            possible_properties = []
            for bay in df_possible_bay['bay'].values[0]:
                bay_breadth = self.df_bay[self.df_bay['bay_name'] == bay]['block_breadth'].values[0]
                bay_height = self.df_bay[self.df_bay['bay_name'] == bay]['block_height'].values[0]
                bay_properties = (bay_breadth, bay_height)
                possible_properties.append(bay_properties)

            idx = np.random.choice(len(possible_properties))
            possible_property = possible_properties[idx]
            breadth = possible_property[0]
            height = possible_property[1]

            df_weight = self.df_bay["block_weight"][(breadth <= self.df_bay["block_breadth"]) &
                                                    (height <= self.df_bay["block_height"])]

            weight = df_weight.max()
            bay_name_df = self.df_bay[(self.df_bay['block_breadth'] == breadth) &
                                   (self.df_bay['block_height'] == height)
                                   & (self.df_bay['block_weight'] == weight)]
            bay_name_series = bay_name_df['bay_name']

        bay_name_series = list(bay_name_series)
        idx = np.random.choice(range(len(bay_name_series)))
        bay_name = bay_name_series[idx]
        if bay_name not in df_possible_bay['bay'].values[0]:
            idx = np.random.choice(range(len(bay_name_series)))
            bay_name = bay_name_series[idx]

        return bay_name, breadth, height, weight

    def generate(self, file_path=None):
        columns = ["block_name", "block_id", "ship_type", "block_type", "process_type", "bay_name",
                   "length", "breadth", "height", "weight", "workload_h1", "workload_h2",
                   "start_date", "duration", "due_date", "pre_buffer", "post_buffer"]

        df_blocks = []

        num_blocks = 0
        start_date = 0

        while True:
            if self.fix_time_horizon:
                if start_date >= self.time_horizon:
                    flag = True
                    del df_blocks[-1]
                else:
                    flag = False
            else:
                if num_blocks == self.num_blocks:
                    flag = True
                else:
                    flag = False

            if flag:
                break

            name = "J-%d" % num_blocks
            id = num_blocks

            # 데이터 생성
            group_code, ship_type, block_type = self.generate_group()
            process_type = self.generate_process(group_code)

            if num_blocks == 0:
                start_date = 0  # 첫번째 착수일은 0으로 고정
            else:
                p = 1 / (1 + self.iat_avg)
                start_date += stats.geom.rvs(p) - 1  # 이전 착수일에 interval을 더하는 형식으로 계산

            if group_code not in ['BC_A', 'BC_S', 'PT_D', 'PT_L', 'PT_R',
                                  'VL_A', 'VL_B', 'VL_D', 'VL_E', 'VL_F', 'VL_S']:

                length = self.generate_property(group_code, process_type, 'L')
                breadth = self.generate_property(group_code, process_type, 'B')
                height = self.generate_property(group_code, process_type, 'H')

                weight = self.generate_weight(group_code, process_type, length, breadth, height)

                bay_name, breadth, height, weight = self.check_eligibility(group_code, process_type, breadth, height, weight)

                workload_h1 = self.generate_workload_h1(group_code, length, breadth)
                workload_h2 = self.generate_workload_h2(group_code, workload_h1)

                duration = self.generate_duration(group_code, workload_h1, workload_h2, weight)

            else: # 샘플링된 그룹에 대한 처리, 한 행의 데이터를 그대로 가져오는 식으로 구현
                df_sample = self.df_sample[self.df_sample['group'] == group_code]
                df_sample = df_sample[df_sample['process_type'] == process_type]
                df_sample.reset_index(inplace=True)
                idx = np.random.choice(range(df_sample.shape[0]))

                length = df_sample.loc[idx, 'length']
                breadth = df_sample.loc[idx, 'breadth']
                height = df_sample.loc[idx, 'height']

                if process_type == 'Final조립':
                    weight = df_sample.loc[idx, 'weight']
                else:
                    weight = self.generate_weight(group_code, process_type, length, breadth, height)

                workload_h1 = df_sample.loc[idx, 'workload_h1']
                workload_h2 = df_sample.loc[idx, 'workload_h2']

                duration = df_sample.loc[idx, 'duration']

            pre_buffer = 5
            post_buffer = self.calculate_buffer(process_type)

            due_date = start_date + duration + post_buffer - 1

            row = [name, id, ship_type, block_type, process_type, bay_name,
                   length, breadth, height, weight, workload_h1, workload_h2,
                   start_date, duration, due_date, pre_buffer, post_buffer]

            df_blocks.append(row)
            num_blocks += 1

        df_blocks = pd.DataFrame(df_blocks, columns=columns)

        if file_path is not None:
            writer = pd.ExcelWriter(file_path)
            df_blocks.to_excel(writer, sheet_name="blocks", index=False)
            writer.close()

        return df_blocks


if __name__ == '__main__':
    import os

    # validation data generation
    block_data_path = "../input/configurations/block_data.xlsx"
    bay_data_path = "../input/configurations/bay_data.xlsx"
    # num_blocks = 50
    time_horizon = 30

    data_src = DataGenerator(block_data_path,
                             bay_data_path,
                             time_horizon=time_horizon,
                             fix_time_horizon=True)

    file_dir = "../input/validation/"
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    iteration = 20
    for i in range(1, iteration + 1):
        file_path = file_dir + "instance-{0}.xlsx".format(i)
        df_blocks = data_src.generate(file_path)