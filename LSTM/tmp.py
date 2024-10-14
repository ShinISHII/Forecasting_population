areas_list=['513375761']
path_weather='../csv_data/amedas_data/kurashiki/output/weather_exo.pickle'
path_warning = '../html/kurashiki/output/warning_list.pickle'


from func_lstm import func_lstm

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
    areas_list = areas_list,
    path_weather = path_weather,
    path_warning = path_warning,
    graphON = False
    )