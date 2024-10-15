import pandas as pd
from func_lstm import func_lstm
import datetime
import os

path_weather='../csv_data/amedas_data/kurashiki/output/weather_exo.pickle'
path_warning = '../html/kurashiki/output/warning_list.pickle'
PATH_mesh = '../csv_data/meshID/ID_Kurashiki_Mabicho_shelter.csv'




# 実行ごとにディレクトリを作成
start_time = datetime.datetime.now()
dir_str = start_time.strftime('%Y%m%d-%H%M-%S')   #20240914-1930-00
directory = f'./pickles/{dir_str}-spl/'
os.makedirs(directory, exist_ok=True)


# KEY_CODEを追加していく
df = pd.read_csv(PATH_mesh)
areas = []
for area in df['KEY_CODE']:
    areas.append(area)
areas_str=''
for k in areas:
    areas_str+=(str(k)+',')
areas_str=areas_str[:-1]  #最後の一文字に余分なカンマが入っているのでその部分を除いて変数を更新
areas_list = areas_str.split(',')

print(f'{len(areas_list)} areas in total: {areas_list}')



for id in areas_list:
    
    print(f'ID: {id} {type(id)}')
    directory_id= directory+id+'_'

    date_rng,df_test_inversed,predicted_date_rng,predicted_test_plot_inversed = func_lstm(
        train_start_date    = '2016-01-01 00:00:00',
        train_end_date      = '2018-06-30 23:59:59',
        test_start_date     = '2018-07-01 00:00:00',
        test_end_date       = '2018-07-11 23:59:59',
        epochs = 5,
        num_of_lstm_layer = 1,
        exDataIsAll = False,
        window_size=24,
        template1=(8, 5),
        areas_list = id,
        directory = directory_id,
        path_weather = path_weather,
        path_warning = path_warning,
        graphON = False
        )
    
    # print()
