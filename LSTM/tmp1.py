from func_lstm import func_lstm
import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
from func_evaluate import calculate_mape,calculate_mse,calculate_rmse

path_weather = '../csv_data/amedas_data/kurashiki/output/weather_exo.pickle'
path_warning = '../html/kurashiki/output/warning_list.pickle'
PATH_mesh = '../csv_data/meshID/ID_Kurashiki_Mabicho_shelter.csv'

import pandas as pd
df = pd.read_csv(PATH_mesh)

# KEY_CODEを追加していく
areas = []
for area in df['KEY_CODE']:
    # print(area)
    areas.append(area)
areas_str=''
for k in areas:
    areas_str+=(str(k)+',')
areas_str=areas_str[:-1]  #最後の一文字に余分なカンマが入っているのでその部分を除いて変数を更新
areas_list = areas_str.split(',')

print(f'{len(areas_list)} areas in total: {areas_list}')



start_time = datetime.datetime.now()
dir_str = start_time.strftime('%Y%m%d-%H%M-%S')  # 20240914-1930-00
directory = f'./pickles/{dir_str}-spl/'
os.makedirs(directory, exist_ok=True)

directory_func = f'{directory}/func/'
os.makedirs(directory_func, exist_ok=True)


# func_lstm の結果をまとめる辞書型変数を定義
results = {'date': {}}  # 日付を格納する辞書を初期化

for idx, id in enumerate(areas_list):
    print(f'ID: {id}', endwith=' ')
    list_id = [id]
    directory_id = directory_func +'/'+ id + '_'

    # func_lstm の出力を取得
    date_obse, pop_obse, date_pred, pop_pred = func_lstm(
        train_start_date='2016-01-01 00:00:00',
        train_end_date='2018-06-30 23:59:59',
        test_start_date='2018-07-01 00:00:00',
        test_end_date='2018-07-11 23:59:59',
        epochs=20,
        num_of_lstm_layer=1,
        exDataIsAll=False,
        window_size=24,
        template1=(8, 5),
        areas_list=list_id,
        directory=directory_id,
        path_weather=path_weather,
        path_warning=path_warning,
        graphON=False
    )

    # 最初のループでのみ観測日付と予測日付を格納
    if idx == 0:
        results['date']['date_obse'] = date_obse
        results['date']['date_pred'] = date_pred

    # id を key として結果を辞書に格納
    results[id] = {
        'pop_obse': pop_obse,
        'pop_pred': pop_pred
    }

# pop_obse と pop_pred の合計を計算
pop_obse_total = [sum(x) for x in zip(*[result['pop_obse'] for result in results.values() if 'pop_obse' in result])]
pop_pred_total = [sum(x) for x in zip(*[result['pop_pred'] for result in results.values() if 'pop_pred' in result])]

# pop_obse_total と pop_pred_total を四捨五入
pop_obse_total = [round(value) for value in pop_obse_total]
pop_pred_total = [round(value) for value in pop_pred_total]

# 結果の表示
# print("Results for each id:")
# for id, data in results.items():
#     print(f"ID: {id}")
#     if 'pop_obse' in data:
#         print(f"  pop_obse: {data['pop_obse']}")
#     if 'pop_pred' in data:
#         print(f"  pop_pred: {data['pop_pred']}")

print("\n\nTotal observed population:",len(pop_obse_total), pop_obse_total)
print("\n\nTotal predicted population:",len(pop_pred_total) ,pop_pred_total)

# pop_obse_total と pop_pred_total のグラフを作成
plt.figure(figsize=(10, 6))
plt.plot(results['date']['date_obse'], pop_obse_total, label='Total Observed Population', marker='o')
plt.plot(results['date']['date_pred'], pop_pred_total, label='Total Predicted Population', marker='x')
plt.title('Forecasted Population')
plt.ylabel('Population')
plt.legend()
plt.grid()
plt.xticks(rotation=45)  # 日付ラベルを回転
plt.tight_layout()

# 画像を保存
plt.savefig(f'{directory}/{dir_str}.png')
plt.show()




