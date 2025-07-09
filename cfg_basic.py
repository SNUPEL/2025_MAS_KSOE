import os
import pandas as pd
import time

class Configure:
    def __init__(self):
        self.data_params = dict()
        self.data_params['n_total'] = 12132
        self.data_params['n_episode_data'] = 50

        self.data_params['VL_D_low'] = 189
        self.data_params['VL_D_high'] = 193

        # 결과 저장
        self.data_params['ymd'] = time.strftime('%Y%m%d')
        self.data_params['hour'] = str(time.localtime().tm_hour)
        self.data_params['minute'] = str(time.localtime().tm_min)
        self.data_params['second'] = str(time.localtime().tm_sec)
        self.data_params["folderpath"] = 'results/{0}_{1}h_{2}m_{3}s'.format(
            self.data_params['ymd'], self.data_params['hour'], self.data_params['minute'], self.data_params['second'])

        self.save_config()

    def save_config(self):
        if not os.path.exists(self.data_params["folderpath"]):
            os.mkdir(self.data_params["folderpath"])

        config_df = pd.json_normalize(self.data_params, sep='_').transpose()
        config_df.to_excel(self.data_params['folderpath'] + '/configuration.xlsx', index=True)


if __name__ == '__main__':
    cfg = Configure()

