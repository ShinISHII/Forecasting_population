# In[1]
from func_lstm import func_lstm
import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
from func_evaluate import calculate_mape,calculate_mse,calculate_rmse
import pandas as pd
import matplotlib.dates as mdates
import sys

path_weather = '../csv_data/amedas_data/kurashiki/output/weather_exo.pickle'
path_warning = '../html/kurashiki/output/warning_list.pickle'
PATH_mesh = '../csv_data/meshID/ID_Kurashiki_Mabicho_shelter.csv'
# exDataIsAll         =  True
# exDataIsAll         =  False
exDataIsAll         = sys.argv[1] == 'True'
train_start_date    = '2016-01-01 00:00:00'
train_end_date      = '2018-06-30 23:59:59'
test_start_date     = '2018-07-01 00:00:00'
test_end_date       = '2018-07-11 23:59:59'
epochs              = 30
window_size         = 24
# num_of_lstm_layer   = 1
num_of_lstm_layer   = int(sys.argv[2])
template1           =(8, 5)
graphON             =True
##############################################################################################################
print(f'Exo: {exDataIsAll,type(exDataIsAll)}')
print(f'Lstm_layers: {num_of_lstm_layer,type(num_of_lstm_layer)}')
print()

df = pd.read_csv(PATH_mesh)
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



# 計測開始時間
from datetime import datetime, timedelta
cal_time_start= datetime.now()

for idx, id in enumerate(areas_list):
    print(f'ID: {id}', end=' ')
    list_id = [id]
    directory_id = directory_func +'/'+ id + '_'

    # func_lstm の出力を取得
    date_obse, pop_obse, date_pred, pop_pred = func_lstm(
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        test_start_date=test_start_date,
        test_end_date=test_end_date,
        epochs=epochs,
        num_of_lstm_layer=num_of_lstm_layer,
        exDataIsAll=exDataIsAll,
        window_size=window_size,
        template1=template1,
        areas_list=list_id,
        directory=directory_id,
        path_weather=path_weather,
        path_warning=path_warning,
        graphON=graphON
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

import matplotlib.pyplot as plt

# FigureとAxesを作成
fig, ax = plt.subplots(figsize=template1)

# 実データと予測データをプロット
ax.plot(results['date']['date_obse'], pop_obse_total, label='Observed', linewidth=1.5, color='gray')
ax.plot(results['date']['date_pred'], pop_pred_total, label='Forecasted by LSTM in each mesh', linewidth=1.5, color='red')

# タイトルとラベルの設定
ax.set_title('Forecasted Population')
ax.set_ylabel('Population')
ax.legend(loc='upper left')
ax.grid()

# 日時のフォーマット設定
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

# フォーマットを月単位に設定
formatter = mdates.DateFormatter('%Y-%b-%d')  # 年-月形式で表示
ax.xaxis.set_major_formatter(formatter)

# レイアウトを調整
plt.tight_layout()

# 画像を保存
plt.savefig(f'{directory}/{dir_str}.png')
plt.show()


mape = calculate_mape(pop_obse_total[-len(pop_pred_total):], pop_pred_total) 
mse = calculate_mse(pop_obse_total[-len(pop_pred_total):], pop_pred_total)
rmse = calculate_rmse(pop_obse_total[-len(pop_pred_total):], pop_pred_total)

print(f'MAPE: {mape:.2f}%')
print(f'MSE : {mse:.2f}')
print(f'RMSE : {rmse:.2f}')

import pickle
import os

# pickle #####################
# 保存する時間情報

### 保存したい複数の変数
a = results
b = pop_obse_total
c = pop_pred_total
d = mape
e = mse
f = rmse


### pickleで保存（書き出し）: 辞書にして複数の変数を保存
pickle_path = os.path.join(directory, f'vars.pickle')
with open(pickle_path, mode='wb') as fo:
    pickle.dump({
        'results': a, 
        'pop_obse_total': b, 
        'pop_pred_total': c, 
        'mape': d, 
        'mse': e, 
        'rmse': f, 

    }, fo)

# 計測終了時間
cal_time_end = datetime.now()

# 経過時間の計算（分単位で）
elapsed_time = cal_time_end - cal_time_start
elapsed_minutes = elapsed_time.total_seconds() / 60


md_file_path = os.path.join(directory, f'report.md')

# In[2]:
with open(md_file_path, 'w') as md_file:
    md_file.write(f"""
# Report for This LSTM model 
Elapsed time: {elapsed_minutes:.2f} min

File created time: {dir_str}

## Result 
<img src="{os.path.basename(f'{directory}/{dir_str}.png')}" width='600'/>

- MAPE\t{mape:.1f}%
- MSE \t{mse:.2f}
- RMSE : {rmse:.2f}

## Pickle path
{pickle_path}

## Mesh path
{PATH_mesh}

## Imput vars

### Exo data:
- {exDataIsAll}

### Period:
- train_start_date    = {train_start_date}
- train_end_date      = {train_end_date}
- test_start_date     = {test_start_date}  
- test_end_date       = {test_end_date}

### LSTM parameter
- window_size\t{window_size}
- epochs\t{epochs}
- n_layers\t{num_of_lstm_layer}

""")
        
# %%
