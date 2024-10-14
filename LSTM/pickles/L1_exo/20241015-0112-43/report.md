
# Report for This LSTM model (Elapsed time: 0.85 min)

File created time: 20241015-0112-43

## Result 
<img src="result_20241015-0112-43_loss.png" width='600'/>
<img src="result_20241015-0112-43_forecast.png" width='600'/>

- MAPE	111.1%
- MSE 	70929.73
- RMSE : 266.33

## Pickle path
./pickles/20241015-0112-43/vars_exoTrue.pickle

## Mesh path
../csv_data/meshID/ID_Kurashiki_Mabicho_shelter.csv

## Imput vars

### Exo data:
- True

### Exogenous data:
- population, prec, temp, wind, 雷注意報, 大雨注意報, 洪水注意報, 強風注意報, 大雨警報, 洪水警報, 暴風警報, 大雨特別警報
 
### Period:
- train_start_date    = 2016-01-01 00:00:00
- train_end_date      = 2018-06-30 23:59:59
- test_start_date     = 2018-07-01 00:00:00  
- test_end_date       = 2018-07-11 23:59:59

### LSTM parameter
- window_size	24
- epochs	30
- ...
- feature_size	12
- n_hidden	64
- n_layers	1
- net

     MyLSTM(
  (lstm): LSTM(12, 64, batch_first=True)
  (fc): Linear(in_features=64, out_features=1, bias=True)
)


