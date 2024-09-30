
# Report for This LSTM model

File created time: 20240920-032741

## MeshID
../csv_data/meshID/ID_Kurashiki_Mabicho_shelter.csv

## Imput vars

### Exo data:
- True

### Exogenous data:
- population, precipitation, temperature, windspeed
 
### Period:
- train_start_date    = 2018-04-01 00:00:00
- train_end_date      = 2018-06-30 23:59:59
- test_start_date     = 2018-07-01 00:00:00  
- test_end_date       = 2018-07-11 23:59:59

### LSTM parameter
- window_size	24
- epochs	15

- feature_size	4
- n_hidden	64
- n_layers	1
- net

     MyLSTM(
  (lstm): LSTM(4, 64, batch_first=True)
  (fc): Linear(in_features=64, out_features=1, bias=True)
)

## Result 
![Plot](result_20240920-032741.png)

- mape	121.7
