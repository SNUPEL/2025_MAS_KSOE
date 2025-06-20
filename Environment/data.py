import pandas as pd
import numpy as np

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

    def generate_process(self):  # 공종 명칭 생성 함수, 공정이 나오는 비율에 맞춰서 데이터 생성
        df_process_count = pd.DataFrame(self.df_revised['공종_명칭'].value_counts())
        df_process_count.reset_index(inplace=True)
        df_process_count['Proportion'] = df_process_count['count'] / df_process_count['count'].sum()
        process_type = np.random.choice(df_process_count['공종_명칭'], p=df_process_count['Proportion'])
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
        columns = ["Block_Name", "Block_ID", "Process_Type", "Ship_Type", "Start_Date", "Duration", "Due_Date",
                   "Workload_H01", "Workload_H02", "Weight", "Length", "Breadth", "Height"]


        df_blocks = []



        for j in range(self.num_blocks):
            name = "J-%d" % j
            id = j

            # 데이터 생성 코드 추가
            process_type = self.generate_process()
            group_code = self.generate_group()
            ship_type = group_code[0]
            block_type = group_code[1]
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