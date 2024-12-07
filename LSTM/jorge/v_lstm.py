import torch.nn as nn
import torch

class MV_LSTM(torch.nn.Module):
    def __init__(self,n_features,seq_length, hidden_dim, num_layers, output_size=1, batch_first = True, bidirectional = False):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.n_hidden = hidden_dim # number of hidden states
        self.n_layers = 1 # because just one per hidden dim
        self.num_layers = num_layers # nr of stacked lstm layers
    
        self.lstm1 = torch.nn.LSTM(input_size = n_features, 
                                 hidden_size = self.n_hidden, #[0]
                                 num_layers = self.n_layers, 
                                 batch_first = self.batch_first,
                                 bidirectional = self.bidirectional)
        # self.lstm2 = torch.nn.LSTM(input_size = self.n_hidden[0], #self.n_hidden, 
        #                           hidden_size = self.n_hidden[1],
        #                           num_layers = self.n_layers, 
        #                           batch_first = self.batch_first,
        #                           bidirectional = self.bidirectional)
        
            # according to pytorch docs LSTM output is 
            # (batch_size,seq_len, num_directions * hidden_size)
            # when considering batch_first = True
            # self.l_linear = torch.nn.Linear(self.n_hidden[1]*self.seq_len, 1)
        if bidirectional:
            self.l_linear = torch.nn.Linear(self.n_hidden*self.seq_len*2, output_size)
        else:
            self.l_linear = torch.nn.Linear(self.n_hidden*self.seq_len, output_size)
        
    
    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        # hidden_state0 = torch.zeros(self.n_layers,batch_size,self.n_hidden[0])
        # cell_state0 = torch.zeros(self.n_layers,batch_size,self.n_hidden[0])
        hidden_state0 = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        cell_state0 = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        self.hidden_layer0 = (hidden_state0, cell_state0)
        
        # hidden_state1 = torch.zeros(self.n_layers,batch_size,self.n_hidden[1])
        # cell_state1 = torch.zeros(self.n_layers,batch_size,self.n_hidden[1])
        # hidden_state1 = torch.zeros(1,batch_size,self.n_hidden[1])
        # cell_state1 = torch.zeros(1,batch_size,self.n_hidden[1])
        # self.hidden_layer1 = (hidden_state1, cell_state1)


    
    def forward(self, x):        
        batch_size, seq_len, _ = x.size() # b, t, c
        # print("batch", batch_size)
        # for i in range(self.seq_len):
        #     lstm_out1, self.hidden_layer0 = self.lstm1(x[:,i,:], self.hidden_layer0)
        #     hidden_state0, cell_state0 = self.lstm1(x[:,i,:], self.hidden_layer0)
        #     hidden_state1, cell_state1 = self.lstm2(hidden_state0, self.hidden_layer1)


        lstm_out1, self.hidden_layer0 = self.lstm1(x) # , self.hidden_layer0) #, self.hidden_layer0)
        # lstm_out2, self.hidden_layer1 = self.lstm2(lstm_out1, self.hidden_layer1)       
        # lstm_out1, self.hidden_layer0 = self.lstm1(x) #, self.hidden_layer0) # (b, c, 2*hidden_dim), 2 if bidirectional
            #lstm_out2, self.hidden_layer1 = self.lstm2(lstm_out1, self.hidden_layer1)
        
            # lstm_out(with batch_first = True) is 
            # (batch_size,seq_len,num_directions * hidden_size)
            # for following linear layer we want to keep batch_size dimension and merge rest       
            # .contiguous() -> solves tensor compatibility error
        # print("lol", x.shape)
        # print(lstm_out1.shape)
        x = lstm_out1.contiguous().view(batch_size,-1) #(b, t*c*hidden_dim*2) 2 if bidirectional
        # print("seeee", x.shape)
        return self.l_linear(x)
        # return lstm_out2
  




