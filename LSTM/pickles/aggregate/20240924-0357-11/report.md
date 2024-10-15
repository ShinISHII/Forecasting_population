
# Report for This LSTM model (Elapsed time: 0.49 min)

File created time: 20240924-0357-11

## Result 
<img src="result_20240924-0357-11_loss.png" width='600'/>
<img src="result_20240924-0357-11_forecast.png" width='600'/>

- MAPE	23.2%
- MSE 	19339.08

## Pickle path
./pickles/20240924-0357-11/vars_exoTrue.pickle

## Mesh path
../csv_data/meshID/ID_Kurashiki_Mabicho_shelter.csv

## Imput vars

### Exo data:
- True

### Exogenous data:
- population, prec, temp, wind, 雷注意報, 大雨注意報, 洪水注意報, 強風注意報, 大雨警報, 洪水警報, 暴風警報, 大雨特別警報
 
### Period:
- train_start_date    = 2021-08-01 00:00:00
- train_end_date      = 2022-09-09 23:59:59
- test_start_date     = 2022-09-15 00:00:00  
- test_end_date       = 2022-09-21 23:59:59

### LSTM parameter
- window_size	24
- epochs	20
- ...
- feature_size	12
- n_hidden	64
- n_layers	2
- net

     MyLSTM(
  (lstm): LSTM(12, 64, num_layers=2, batch_first=True)
  (fc): Linear(in_features=64, out_features=1, bias=True)
)


