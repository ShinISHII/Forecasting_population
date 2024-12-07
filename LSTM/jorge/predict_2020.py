
"""
In this script, the trained models are used to make predictions.
It is used to predict 2020 for validation.
"""


import numpy as np
import os
import torch
from tqdm import tqdm
from torch.autograd import Variable
from model.v_convlstm import ConvLSTM
from model.bi_convlstm import ConvBLSTM
# from model.v_convgru import ConvGRU
from model.v_lstm import MV_LSTM
# from model.v_gru import GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import matplotlib.pyplot as plt
import tifffile
from utilis.popFunct import filter_fft, histo_thresh


proj_dir = "/Users/jorgemorales/Desktop/Jorge/2024/ModelC/"

# define config
config = {
        "l1": 64, 
        "l2": 'na', 
        "lr": 0.0012, 
        "batch_size": 2, 
        "epochs": 15, #50
        "model_n" : '02-20_3y',
        "save" : True,
        # "model": 'BiLSTM', # 'ConvLSTM', 'LSTM', 'BiConvLSTM', ('linear_reg', 'multivariate_reg',' 'random_forest_reg')
        "factors" : 'pop', # 'all', 'static', 'pop'
        "run" : 'run3'
    }


def evaluate(gt, pred):
    mae = metrics.mean_absolute_error(gt, pred)
    rmse = metrics.mean_squared_error(gt, pred, squared = False)    
    return mae, rmse


def get_subsample_centroids(img, img_size=50):
    h_total = img.shape[-2]
    w_total = img.shape[-1]
    h_step = int(h_total // img_size * 1.5)
    w_step = int(w_total // img_size * 1.5)
    x_list = np.linspace(img_size//2, h_total-img_size//2, num = h_step)
    y_list = np.linspace(img_size//2, w_total -img_size//2, num= w_step)

    new_x_list = []
    new_y_list = []

    for i in x_list:
        for j in y_list:
            new_x_list.append(int(i))
            new_y_list.append(int(j))
    return new_x_list, new_y_list



def main(*kwargs):
    conv = False if config['model'] in ['LSTM' , 'BiLSTM'] else True
    if conv == False: # LSTM and GRU
        config['batch_size'] = 1
        
        
    save_name = '{}_{}_{}/lr{}_bs{}_1l{}_2l{}/{}/'.format(config["model"], config["model_n"], config["factors"], config["lr"], config["batch_size"], config["l1"], config["l2"], config["run"])        


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    
    ori_data_dir = proj_dir + 'data/ori_data/input_all.npy'
    ori_data = np.load(ori_data_dir)# .transpose((1, 0, 2, 3))
    
    if config['model_n'] == '02-20_3y':
        #[7,10,13,16,19]
        valid_input = ori_data[[4,7,10,13,16], :, :, :] # years 2005-2020, 3y interval
          
    elif config['model_n'] == '04-20_4y':
        valid_input = ori_data[[11,15,19], :, :, :] # years 2004-2020, 4y interval
        # valid_input = valid_input[:,[1,3,4,5,6],:,:]              # static input features
        
    elif config['model_n'] == '02-20_2y':
        valid_input = ori_data[[5,7,9,11,13,15,17,19], :, :, :] # years 2006-2020, 2y interval
        # valid_input = valid_input[:,[1,3,4,5,6],:,:]                        # static input features
        
    elif config['model_n'] == '01-20_1y':
        valid_input = ori_data[2:, :, :, :] # years 2002-2020, 1y interval
        # valid_input = valid_input[:,[1,3,4,5,6],:,:]  # static input features
    
    
    
    if config['factors'] == 'all':
        valid_input = valid_input[:,1:,:,:]  # all input features except multiclass lc
    elif config['factors'] == 'static':
        valid_input = valid_input[:,[1,3,4,5,6],:,:]  # static input features
    elif config['factors'] == 'pop':
        valid_input = valid_input[:, 1, :, :] # population data only
    #    
    print("valid input", valid_input.shape, valid_input.min(), valid_input.max())

    gt = ori_data[-1, 1, :, :] # last year, population
    
    if conv == False: # LSTM and GRU
        seq_length = valid_input.shape[0]
        
    input_channel = 10 if config['factors'] == 'all' else 5 if config['factors'] == 'static' else 1
            
    
    # load checkpoint of last epoch from training            
    dir_checkpoint = proj_dir + 'data/ckpts/' + save_name + 'CP_epoch{}.pth'.format(config["epochs"]-1)        
    
    
    # slice the image for prediction - stitch together afterwards
    x_list, y_list = get_subsample_centroids(valid_input, img_size=256)
    
    sub_img_list = []
    for x, y in zip(x_list, y_list):
        if config['factors'] == 'pop':
            sub_img = valid_input[:, np.newaxis, x - 128:x + 128, y - 128:y + 128]
        else:
            sub_img = valid_input[:, :, x - 128:x + 128, y - 128:y + 128]
        sub_img_list.append(sub_img)
    
    pred_img_list = []
    
    with torch.no_grad():
        for test_img in tqdm(sub_img_list):
  
            if config["model"] == 'ConvLSTM':       
                net = ConvLSTM(input_dim = input_channel,
                               hidden_dim= 64, #[config['l1'], 1], 
                               kernel_size=(3,3), num_layers = 1, # (3,3), num_layers = 2, 
                               batch_first=True, return_all_layers=False)
            
            elif config["model"] == 'BiConvLSTM':
                net = ConvBLSTM(input_dim = input_channel,
                               hidden_dim=config['l1'],
                               kernel_size=3,
                               batch_first=True, return_all_layers=False)
    
            elif config["model"] == 'LSTM':
                  net = MV_LSTM(n_features = input_channel,
                                seq_length = seq_length,
                                hidden_dim = config['l1'],
                                num_layers = 1,
                                batch_first = True,
                                bidirectional = False)
         
            elif config["model"] == 'BiLSTM':
                  net = MV_LSTM(n_features = input_channel,
                                seq_length = seq_length,
                                hidden_dim = config['l1'],
                                num_layers = 1,
                                batch_first = True,
                                bidirectional = True)
                  
            # elif config["model"] == 'ConvGRU':
            #     net = ConvGRU(input_dim = input_channel,
            #                   hidden_dim = [config['l1'],1],
            #                   kernel_size=(3,3), 
            #                   num_layers = 2,
            #                   batch_first = True, return_all_layers=False)
                
            net.to(device)
            net.load_state_dict(torch.load(dir_checkpoint))
            
            
            if conv == False: # LSTM and GRU
                test_img = test_img.reshape(test_img.shape[0], test_img.shape[1], test_img.shape[-2]*test_img.shape[-1]) # (t,c, w*h)
                # print(test_img.shape,test_img.min(), test_img.max())
                test_img = np.moveaxis(test_img, 2, 0) # (w*h, t, c)
                test_img = torch.from_numpy(test_img.copy()).to(device=device, dtype=torch.float32) # (w*h, t, c)
                print("holli", test_img.shape,test_img.min(), test_img.max())
                # net.init_hidden(test_img.shape[0])
                pred_img = net(test_img) 
                print("before last pred", pred_img.shape)
                pred_img = pred_img[:, 0] # take last year prediction
                print(pred_img.shape, pred_img.min(), pred_img.max())
                pred_img_list.append(pred_img.cpu().numpy().reshape(256, 256))
            
            
            else: # convLSTM
                test_img = Variable(torch.from_numpy(test_img.copy())).unsqueeze(0).to(device=device,
                                                                                       dtype=torch.float32)
                output_list = net(test_img) 
                pred_img = output_list[0].squeeze()
                pred_img = pred_img[-1,:,:] # take last year prediction
                pred_img_list.append(pred_img.cpu().numpy())
               
    
    pred_msk = np.zeros((valid_input.shape[-2], valid_input.shape[-1]))
    
    h = 0
    # x_list, y_list = get_subsample_centroids(valid_input, img_size=256)
    for x, y in zip(x_list, y_list):
        if x == np.min(x_list) or x == np.max(x_list) or y == np.min(y_list) or y == np.max(y_list):
            pred_msk[x - 128:x + 128, y - 128:y + 128] = pred_img_list[h]
            h += 1
        else:
            # pred_msk[x - 120:x + 120, y - 120:y + 120] = pred_img_list[h][8:248,8:248]
            pred_msk[x - 106:x + 106, y - 106:y + 106] = pred_img_list[h][22:234,22:234]
            h += 1
    
    
    val_mae, val_rmse = evaluate(gt, pred_msk)
    print('mae: ', val_mae)
    print('rmse: ', val_rmse)
    plt.imshow(pred_msk)
    #
    print(pred_msk.min(), pred_msk.max(), pred_msk.mean())
    
    
    # rescale to actual pop values
    ori_unnormed = np.load(proj_dir + 'data/ori_data/input_all_unnormed.npy')
    pop_unnormed = ori_unnormed[:, 1, :, :]
    #
    print(pop_unnormed.shape)
    print(pop_unnormed.min(), pop_unnormed.max(), pop_unnormed.mean())
    #
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(pop_unnormed.reshape(-1, 1))
    pop = scaler.inverse_transform(pred_msk.reshape(-1,1)).reshape(pred_msk.shape[-2], pred_msk.shape[-1])
    #
    save_path = proj_dir + "data/test/" + save_name
    os.makedirs(save_path, exist_ok=True)
    # #
    # pop_filtered_norm = filter_fft(pred_msk, thresh=0.3, plots=False)            ##########
    # pop_filtered_rescaled  = filter_fft(pop, thresh=0.3, ws=save_path)          ##########
    pop_filtered_norm = histo_thresh(pred_msk, h=0.1, ws=save_path, save=False)            ##########
    pop_filtered_rescaled  = histo_thresh(pop, h=0.01, bin= 20,  ws=save_path)          ##########
    #
    np.save(save_path + 'pred_msk_eval_normed.npy', pred_msk)
    np.save(save_path + 'pred_msk_eval_rescaled.npy', pop)
    #
    np.save(save_path + 'pred_msk_filtered_normed.npy', pop_filtered_norm)       ##########
    np.save(save_path + 'pred_msk_filtered_rescaled.npy', pop_filtered_rescaled) ##########
    #
    plt.savefig(save_path + 'pred_msk_eval.png')
    
    
    tifffile.imwrite(save_path + 'pred_msk_normed.tif', pred_msk)
    tifffile.imwrite(save_path + 'pred_msk_rescaled.tif', pop)
    tifffile.imwrite(save_path + 'pred_msk_filtered.tif', pop_filtered_rescaled)
    


# run for all models    
all_models = ['LSTM']     # ['LSTM', 'BiLSTM', 'ConvLSTM' 'BiConvLSTM' ] 
all_factors = ['pop']       # ['pop', 'static', 'all']
all_modeln = ['02-20_3y']   # ['01-20_1y', '02-20_2y', '02-20_3y', '04-20_4y']
runs = ['run3'] # ['run1', 'run2', 'run3', 'run4', 'run5']


for m in all_models:
    for n in all_modeln:
        for r in runs:
            config['model'] = m
            config['model_n'] = n
            config['run'] = r
            main(config)

