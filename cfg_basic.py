import os
import pandas as pd
import time

class Configure:
    def __init__(self):
        self.data_params = dict()
        self.data_params['data_revised_filepath'] = './data/블록-계획데이터(예제)_수정_선종블록 추가.xlsx'      # 사용 데이터파일
        self.data_params['data_group'] = './data/선종블록_공정수.xlsx'        # 그룹별 카운트

        # 항목별 모델 변수
        self.data_params['model_for_property'] = './data/그룹별 제원 변수.xlsx'
        self.data_params['model_for_W'] = './data/그룹별 W 변수.xlsx'
        self.data_params['model_for_H01'] = './data/그룹별 H01 변수.xlsx'
        self.data_params['model_for_H02'] = './data/그룹별 H02 변수.xlsx'
        self.data_params['model_for_duration'] = './data/그룹별 duration 변수.xlsx'

        # 착수일 간격, 후버퍼 보정을 위한 확률
        self.data_params['p_for_interval'] = 0.8952
        self.data_params['p_for_buffer'] = 0.3886

        # 샘플링되는 그룹들(데이터 수가 10개 이하인 그룹들)
        self.data_params['group_sampling'] = ['BC_A', 'BC_S', 'PT_D', 'PT_L', 'PT_R', 'VL_A', 'VL_B', 'VL_D', 'VL_E', 'VL_F', 'VL_S']


        self.data_params['n_total'] = 6938
        self.data_params['n_episode_data'] = 50

        self.data_params['weight_max_limit_ratio'] = 0.7
        # self.data_params['VL_D_low'] = 189
        # self.data_params['VL_D_high'] = 193

        # 결과 저장
        self.data_params['ymd'] = time.strftime('%Y%m%d')
        self.data_params['hour'] = str(time.localtime().tm_hour)
        self.data_params['minute'] = str(time.localtime().tm_min)
        self.data_params['second'] = str(time.localtime().tm_sec)
        self.data_params["folderpath"] = 'results/{0}_{1}h_{2}m_{3}s'.format(
            self.data_params['ymd'], self.data_params['hour'], self.data_params['minute'], self.data_params['second'])

        # self.save_config()

    def save_config(self):
        if not os.path.exists(self.data_params["folderpath"]):
            os.mkdir(self.data_params["folderpath"])

        config_df = pd.json_normalize(self.data_params, sep='_').transpose()
        config_df.to_excel(self.data_params['folderpath'] + '/configuration.xlsx', index=True)


if __name__ == '__main__':
    cfg = Configure()

