def func_lstm(train_start_date,
              train_end_date,
              test_start_date,
              test_end_date,
              epochs,
              num_of_lstm_layer,
              exDataIsAll,
              areas_list,
              path_weather,
              path_warning,
              directory,
              graphON = False,
              template1=(8, 5),
              window_size=24,
              ):
    import pandas as pd
    import psycopg2
    import datetime
    import matplotlib.pyplot as plt
    import pickle
    import warnings
    plt.clf()

    # UserWarningの警告を無視する
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    from datetime import datetime, timedelta
    # 計測開始時間
    cal_time_start= datetime.now()


# 日付加工：外部変数を読みたるため
    split_date = test_start_date

    date_st = train_start_date[0:4]+train_start_date[5:7]+train_start_date[8:10]+train_start_date[11:13]+train_start_date[14:16]
    date_en = test_end_date[0:4]+test_end_date[5:7]+test_end_date[8:10]+test_end_date[11:13]+test_end_date[14:16]

    # print(train_start_date,'\t→\t',test_end_date)
    # print(date_st,'\t\t→\t',date_en)

# データベースの接続情報
    connection_config = {
        'user': 'shin',
        'password': 'shin_password',
        'port': '5432',
        'database': 'mobaku_base',
        'host': '10.33.230.198'
    }
    connection = psycopg2.connect(**connection_config)

    with connection.cursor() as cursor:
        cursor.execute("SET pg_strom.enabled=off;")
        connection.commit()


    # データを取得（datetimeごとのpopulation合計を計算）
    sql_query = f"""
    SELECT datetime, SUM(population) AS population
    FROM population_00000
    WHERE mesh_id IN ({','.join(areas_list)})
    AND datetime BETWEEN '{train_start_date}' AND '{test_end_date}'
    GROUP BY datetime
    ORDER BY datetime;
    """


    df_pop = pd.read_sql(sql=sql_query, con=connection)

    # pd.date_rangeを使用して、指定された期間のすべての時間ステップを含むDataFrameを作成
    expected_index = pd.date_range(start=train_start_date, end=test_end_date, freq='H')
    df_expected = pd.DataFrame(index=expected_index)

    # 取得したデータをdatetime列をインデックスに設定してマージ
    df_pop['datetime'] = pd.to_datetime(df_pop['datetime'])
    df_pop.set_index('datetime', inplace=True)

    # expected_indexを使って、すべての時間が揃うようにmerge（欠損はNaNで埋める）
    df_pop = df_expected.merge(df_pop, left_index=True, right_index=True, how='left')

    # 欠損値を一つ前の値で埋める（必要に応じて他の方法で埋めることも可能）
    df_pop['population'].ffill(inplace=True)

    # print(df_pop)

    # ## Import exogenous data (weather)
    df_weather_all=pd.read_pickle(path_weather)
    df_weather=df_weather_all[train_start_date:test_end_date]


    # ## Import exogenous data (warnings/advisories)
    # pickleで保存したファイルを読み込み
    with open(path_warning, mode='rb') as fi:
        data = pickle.load(fi)

    df_warnings_tmp = data['combined_df']

    # データを指定した期間でフィルタリング
    df_warnings = df_warnings_tmp.loc[train_start_date:test_end_date]

    timeLine=df_warnings.index
    # print('timeline')
    # print(timeLine)
    # print()

    # 日本語フォントを可能にするアイテム
    from matplotlib import rcParams
    rcParams['font.family'] = 'Noto Sans CJK JP'

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates



# 呼び出した外生変数を統合したdataframeを作成
    df_timeLine = pd.DataFrame(timeLine)
    # print(df_timeLine)
    # 他のデータをリセットインデックスで結合
    df_pop_reset = df_pop[['population']].reset_index(drop=True)
    df_weather_reset = df_weather[['prec', 'temp', 'wind']].reset_index(drop=True)
    df_warnings_reset = df_warnings.reset_index(drop=True)

    # print(df_pop_reset,df_weather_reset,df_warnings_reset)
    # 横方向に結合
    df_ex = pd.concat([df_timeLine, df_pop_reset, df_weather_reset,df_warnings_reset], axis=1)

    # 時系列をインデックスに設定
    df_ex = df_ex.set_index('time')
    # print('df_ex')
    # print(df_ex)
    # print()


# データの欠損値の補完
    for name_var in df_ex.columns:
        nan_locations = df_ex[df_ex[name_var].isna()]
        # NaNが見つかった場合
        if not nan_locations.empty:
            if not name_var == 'population':
                # NaNを一つ前のデータで上書きする
                df_ex.loc[:, name_var] = df_ex[name_var].ffill()
                print(f"{name_var} の以下にNaN（欠損値）あり！　以下をffillにて修正した:")
                print(nan_locations)
                print()
            else:
                raise ValueError("Error:人口データの損失が生じた")




    # # LSTM
    # ライブラリのインポート
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    import torch

    from torch import nn,optim
    from torch.utils.data import DataLoader, TensorDataset
    from torchinfo import summary
    from torch.autograd import Variable


    # ## Normalize dataframe values
# data split
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    if exDataIsAll:
        df_ex_standard =scaler.fit_transform(df_ex)
    else:
        df_ex_standard = scaler.fit_transform(df_ex[['population']])

    # testデータの正規化を戻すためのmax,minを抽出
    max_avetem= max(df_ex['population'])
    min_avetem= min(df_ex['population'])

    # trainのデータ行数を求める．
    array_true_false = df_ex.index < split_date
    # print(array_true_false)
    true_count=np.sum(array_true_false)
    # print(true_count)  # <- trainデータ個数

    # train / test に分割
    df_train = df_ex_standard[:true_count]
    df_test = df_ex_standard[true_count:]

    n_dim = df_train.shape[1]


# ## Data slice
    import numpy as np

    window_size = window_size # ウィンドウサイズ（24時間）
    forecast_steps = 1  # xx時間後を予測
    n_data = len(df_ex) - window_size - forecast_steps + 1

    # 訓練データとテストデータのサイズを変更
    n_train = len(df_train) - window_size - forecast_steps + 1
    n_test = len(df_test) - window_size - forecast_steps + 1

    # 正解データを準備________________________________________
    train = np.zeros((n_train, window_size, n_dim))
    train_labels = np.zeros((n_train, n_dim))
    for i in range(n_train):
        train[i] = df_train[i:i+window_size]
        train_labels[i] = df_train[i + window_size + forecast_steps - 1]  # 24時間後のデータを取得

    # テストデータを準備______________________________________
    test = np.zeros((n_test, window_size, n_dim))
    test_labels = np.zeros((n_test, n_dim))
    for i in range(n_test):
        test[i] = df_test[i:i+window_size]
        test_labels[i] = df_test[i + window_size + forecast_steps - 1]  # 24時間後のデータを取得

    train_labels = train_labels[:, 0]  # 最初の列のみを使用



# ## Define LSTM model
    train_torch = torch.tensor(train, dtype=torch.float)
    labels = torch.tensor(train_labels, dtype=torch.float)
    dataset = TensorDataset(train_torch, labels)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    train_torch = torch.tensor(train, dtype=torch.float)
    labels = torch.tensor(train_labels, dtype=torch.float)
    dataset = torch.utils.data.TensorDataset(train_torch, labels)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)



    if num_of_lstm_layer==1:
        # ##1層LSTM
        # 多変量を入力して、１変数の予測結果を返すLSTNモデル.
        class MyLSTM(nn.Module):
            def __init__(self, feature_size, hidden_dim, n_layers):
                super(MyLSTM, self).__init__()

                self.feature_size = feature_size
                self.hidden_dim = hidden_dim
                self.n_layers = n_layers
                self.n_output = 1

                self.lstm = nn.LSTM(feature_size, hidden_dim, num_layers=n_layers, batch_first=True)
                self.fc = nn.Linear(hidden_dim, self.n_output)

            def forward(self, x):
                # hidden state
                h_0 = Variable(torch.zeros(self.n_layers, x.size(0), self.hidden_dim))
                # cell state
                c_0 = Variable(torch.zeros(self.n_layers, x.size(0), self.hidden_dim))
                
                output, (hn, cn) = self.lstm(x, (h_0, c_0)) # (input, hidden, and internal state)
                hn = hn.view(-1, self.hidden_dim) 
                y = self.fc(hn)
                y = y.reshape(self.n_output, -1)

                return y.squeeze()




    if num_of_lstm_layer>=2:
        ##2層LSTM
        class MyLSTM(nn.Module):
            def __init__(self, feature_size, hidden_dim, n_layers=2):  # デフォルトで2層
                super(MyLSTM, self).__init__()

                self.feature_size = feature_size
                self.hidden_dim = hidden_dim
                self.n_layers = n_layers
                self.n_output = 1

                # 2層のLSTM
                self.lstm = nn.LSTM(feature_size, hidden_dim, num_layers=n_layers, batch_first=True)
                self.fc = nn.Linear(hidden_dim, self.n_output)

            def forward(self, x):
                # hidden state
                h_0 = Variable(torch.zeros(self.n_layers, x.size(0), self.hidden_dim))
                # cell state
                c_0 = Variable(torch.zeros(self.n_layers, x.size(0), self.hidden_dim))
                
                # LSTMの2層を通過
                output, (hn, cn) = self.lstm(x, (h_0, c_0))

                # hnの形状を(batch_size, hidden_dim)に変換
                hn = hn[-1]  # 最終タイムステップの出力を使用

                # 全結合層を通して最終出力を取得
                y = self.fc(hn)

                return y.squeeze()  # (batch_size,) にする

    if exDataIsAll:
        feature_size  = 12
    else:
        feature_size  = 1

    n_hidden  = 64
    n_layers  = num_of_lstm_layer

    net = MyLSTM(feature_size, n_hidden, n_layers)
    # print(f'LSTM_layer: {num_of_lstm_layer}')
    # summary(net)


    func_loss = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
        ## 最適化を図るための仕組みとして，Adamを使う．
    loss_history = []
    device = torch.device("cuda:0" if torch.cuda. is_available() else "cpu")
    epochs = epochs
    # print(f'epochs: {epochs}')
    net.to(device)

    for i in range(epochs+1):
        net.train()
        tmp_loss = 0.0
        for j, (x, t) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            y = net(x) 
            y = y.to('cpu')
            loss = func_loss(y, t)
            loss.backward()
            optimizer.step() 
            tmp_loss += loss.item()
        tmp_loss /= j+1
        loss_history.append(tmp_loss)
        # print('Epoch:', i, ', Loss_Train:', tmp_loss)

    plt.plot(range(len(loss_history)), loss_history, label='train')
    plt.legend()

    plt.xlabel("Epochs")
    plt.ylabel("Loss [MSE]")

    # 主グリッドの設定
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    # 補助グリッドの設定
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')

    plt.title('Learning Curve') #学習曲線というらしいわ
    graph_loss=f'{directory}result_loss.png'
    plt.savefig(graph_loss,dpi=300)


    # observed data を用いて予測をする


    predicted_test_plot = []
    net.eval()

    for k in range(n_test):
        x = torch.tensor(test[k])
        x = x.reshape(1, window_size, feature_size)
        x = x.to(device).float()
        y = net(x)
        y = y.to('cpu')
        predicted_test_plot.append(y.item())  # y[0].item() を y.item() に変更

    

    # 正規化を戻す
    #Observed
    df_test_inversed = np.array(df_test[:,0]) * (max_avetem-min_avetem) + min_avetem

    #Predicted
    predicted_test_plot_inversed = np.array(predicted_test_plot) * (max_avetem-min_avetem) + min_avetem

    # 日時データを生成します（1時間おきのデータを使用）
    date_rng = pd.date_range(start=test_start_date, periods=len(df_test_inversed), freq='h')
    predicted_date_rng = date_rng[-len(predicted_test_plot_inversed):]



    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np

    # プロット
    fig, ax = plt.subplots(figsize=template1)  # サイズを指定する

    # データのプロット
    ax.plot(date_rng, df_test_inversed, label='Observed', linewidth=1.5, color='gray')
    ax.plot(predicted_date_rng, predicted_test_plot_inversed, label='Forecasted by LSTM', linewidth=1.5, color='red')
    ax.legend(loc='upper left')

    # 日時のフォーマット設定
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax.grid()

    # フォーマットを月単位に設定
    formatter = mdates.DateFormatter('%Y-%b-%d')  # 年-月形式で表示
    ax.xaxis.set_major_formatter(formatter)

    # x軸のラベルを90度回転させる
    ax.tick_params(axis='x', rotation=90)

    # 0度のラベルを削除
    ax.tick_params(axis='x', which='both', bottom=False, top=False)

    if exDataIsAll:
        add_title_name  = ' with Exogenous variables'
    else:
        add_title_name  = ' without exogenous variables'

    plt.title('Forecasted population' + add_title_name)
    plt.ylabel('Population')
    graph_forecast = f'{directory}result_forecast.png'
    plt.savefig(graph_forecast, dpi=300)

    if graphON:
        plt.show()
    else:
        plt.close()


    # 計測終了時間
    cal_time_end = datetime.now()

    # 経過時間の計算（分単位で）
    elapsed_time = cal_time_end - cal_time_start
    elapsed_minutes = elapsed_time.total_seconds() / 60

    # 結果を出力
    print(f"\telapsed time: {elapsed_minutes:.2f} min")

    return date_rng,df_test_inversed,predicted_date_rng,predicted_test_plot_inversed,