
# Report for This LSTM model (Elapsed time: 2.02 min)

File created time: 20240925-192059

## Result 
<img src="result_20240925-192059_loss.png" width='600'/>
<img src="result_20240925-192059_forecast.png" width='600'/>
<img src="result_20240925-192059_forecast_september_2023.png" width='600'/>

- MAPE	12.2%
- MSE 	nan

## Pickle path
./pickles/20240925-192059/vars_exoTrue.pickle

## Imput vars

### Exo data:
- True

### Exogenous data:
- population, prec, temp, wind, 雷注意報, 大雨注意報, 洪水注意報, 強風注意報, 大雨警報, 洪水警報, 暴風警報, 大雨特別警報
 
### Period:
- train_start_date    = 2016-01-01 00:00:00
- train_end_date      = 2022-12-31 23:59:59
- test_start_date     = 2023-01-01 00:00:00  
- test_end_date       = 2023-12-31 23:59:59

### LSTM parameter
- window_size	24
- epochs	15
- ...
- feature_size	12
- n_hidden	64
- n_layers	2
- net

     MyLSTM(
  (lstm): LSTM(12, 64, num_layers=2, batch_first=True)
  (fc): Linear(in_features=64, out_features=1, bias=True)
)


