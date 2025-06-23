import pandas as pd
import numpy as np

class DataGenerator:
    def __init__(self,
                 num_blocks=50):

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
        self.df_revised.reset_index(inplace=True)

        # 공정 개수에 대한 데이터프레임
        # self.df_process_count = pd.DataFrame(self.df_revised['공종_명칭'].value_counts())
        # self.df_process_count.reset_index(inplace=True)
        # self.df_process_count['Proportion'] = self.df_process_count['count'] / self.df_process_count['count'].sum()

        # 그룹 개수에 대한 데이터프레임
        self.df_revised_for_group = self.df_revised
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

        return (ship_type, block_type)

    def generate(self, file_path=None):
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
            start_date =
            duration =
            due_date =
            workload_h01 =
            workload_h02 =
            weight =
            length =
            breadth =
            height =

            row = [name, id, process_type, ship_type, block_type, start_date, duration, due_date,
                   workload_h01, workload_h02, weight, length, breadth, height]

            df_blocks.append(row)

        df_blocks = pd.DataFrame(df_blocks, columns=columns)

        if file_path is not None:
            writer = pd.ExcelWriter(file_path)
            df_blocks.to_excel(writer, sheet_name="blocks", index=False)
            writer.close()

        return df_blocks