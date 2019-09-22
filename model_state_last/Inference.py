#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import numpy  as np


# In[2]:


# DEVICE_ID = 0
# DEVICE = torch.device('cuda:%d' % DEVICE_ID)

DEVICE = torch.device('cpu')


# In[3]:


def string_number(day):
    return (str(int (day/100)%10) +
            str(int (day/10)%10) +
            str(int (day/1)%10))

def create_mining_stuff(path):
    ret = []
    for i in range(9):
        ret.append(torch.load(path + string_day_nomber(i)))
    return ret


# In[4]:


pump_data = pd.read_csv('pumping.csv', sep = ';', names = ['x', 'y'])
# print (pump_data)


# In[5]:


def make_int_coords(df):

    df['int_x'] = df.x
    df['int_y'] = df.y

    df.int_x -= np.mean(df.int_x)
    df.int_x += 7200

    df.int_y -= np.mean(df.int_y)
    df.int_y += 7200

    num_cells = 50 
    Max_val = 14000 
    df.int_x = df.int_x * num_cells / Max_val 
    df.int_y = df.int_y * num_cells / Max_val

    df.int_x = df.int_x.astype('int')
    df.int_y = df.int_y.astype('int')


# In[6]:


make_int_coords(pump_data)
pump_data


# In[7]:


coord_tensor = torch.zeros([len(pump_data), 2], dtype=torch.int32)

i = 0
for index, row in pump_data.iterrows():
    coord_tensor[i][0] = int(row['int_x'])
    coord_tensor[i][1] = int(row['int_y'])
    i += 1
    
mining_coordinates = [coord_tensor for i in range(9)]


# In[8]:


# print (mining_coordinates[0])


# In[9]:


inject_data = pd.read_csv('injection.csv', sep = ';', names = ['x', 'y', 'water'])
make_int_coords(inject_data)
# inject_data.int_x.astype()
# print (inject_data)


# In[10]:


injection = torch.zeros([9, 1, 50, 50], dtype=torch.float)

for i in range(9):
    for index, row in inject_data.iterrows():
#         print (injection[i][0][ row['int_x'] ][ row['int_y'] ])
#         print (row['water'])
        injection[i][0][ row['int_x'] ][ row['int_y'] ] += float(row['water'])


# In[ ]:





# In[11]:


# injection = torch.load("Data/test_tensor_1div7")
# mining_coordinates = create_mining_stuff("Data/test_mining_int_coordinates_1div7/input_tensor_int")


# In[12]:


# print (injection.shape)
# print (len(mining_coordinates))


# In[13]:


N_FEATURES_INJECTION = injection.shape[1]
N_CELLS_HOR = injection.shape[2]
N_CELLS_VER = injection.shape[3]


# In[14]:


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        
        self.CONV  = nn.Conv2d    (in_channels,
                                   out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   bias=False)             # think about it later
        
        self.BNORM =nn.BatchNorm2d(out_channels,
                                   eps=1e-05,
                                   momentum=0.1,
                                   affine=False)
#         self.RELU  = nn.ReLU ()
        
#         self.MAXPOOL = nn.MaxPool2d(3,
#                                     stride=1,
#                                     padding=1,
#                                     dilation=1)
        
    def forward(self, x):
        #print ('sizeof(x) = ', x.size())
        #print ('sizeof(xprev) = ', xprev.size())    
        
        x = self.CONV   (x)
        x = self.BNORM  (x)
#         x = self.RELU   (x)
#         x = self.MAXPOOL(x)
        
        return x


# In[15]:


class MapToAmount (nn.Module):
    def __init__(self, kernel_radius=1):
        super(MapToAmount, self).__init__()
        
        self.n_features = int(2 * kernel_radius - 1) ** 2
        self.linear = nn.Linear(self.n_features, 1)
        
    def forward(self, mapa):
#         print (mapa.shape)
#         print (self.n_features)
#         print (mapa.view(self.n_features).shape)
        return self.linear(mapa.view(self.n_features))  


# In[16]:


class LSTMCell (nn.Module):
    
    def __init__ (self,
                  n_features_in,
                  n_features_out,
                  embedding_size=16,
                  hidden_state_size=16,
                  output_lin_radius=1):
        super(self.__class__,self).__init__()
        
        self.out_rad  = output_lin_radius
        self.emb_size = embedding_size
        self.hid_size = hidden_state_size
        
#         self.embedding = ConvBlock (1, self.emb_size, kernel_size=3)
        self.embedding = nn.Sequential(ConvBlock(n_features_in,
                                                 self.emb_size,
                                                 3),
                                       nn.ReLU(),
                                       ConvBlock(self.emb_size,
                                                 self.emb_size,
                                                 3))
        
        self.f_t = nn.Sequential (ConvBlock(self.hid_size + self.emb_size,
                                            self.hid_size,
                                            3),
                                  nn.Sigmoid())
        self.i_t = nn.Sequential (ConvBlock(self.hid_size + self.emb_size,
                                            self.hid_size,
                                            3),
                                  nn.Sigmoid())
        self.c_t = nn.Sequential (ConvBlock(self.hid_size + self.emb_size,
                                            self.hid_size,
                                            3),
                                  nn.Tanh())
        self.o_t = nn.Sequential (ConvBlock(self.hid_size + self.emb_size,
                                            self.hid_size,
                                            3),
                                  nn.Sigmoid())
        
        
        #===========Output stuff============================================
        self.hidden_to_result = nn.Sequential (ConvBlock (hidden_state_size, 
                                                          hidden_state_size, 
                                                          kernel_size=3),
                                               nn.ReLU   (),
                                               # TODO BatchNorm
                                               ConvBlock (hidden_state_size,
                                                          n_features_out,
                                                          kernel_size=3,
                                                          padding=4))
        
        self.  oil_result = MapToAmount(kernel_radius=output_lin_radius)
        self.water_result = MapToAmount(kernel_radius=output_lin_radius)
        self.  gas_result = MapToAmount(kernel_radius=output_lin_radius)
        
        
    def forward (self, x, prev_state, holes_coordinates):
        (prev_c, prev_h) = prev_state
        prev_c = prev_c.to('cpu')
        prev_h = prev_h.to('cpu')
        x_emb = self.embedding(x)
        
#         print (prev_h.device)
#         print (x_emb.device)
        x_and_h = torch.cat([prev_h, x_emb], dim=1)
        
        f_i = self.f_t(x_and_h)
        i_i = self.i_t(x_and_h)
        c_i = self.c_t(x_and_h)
        o_i = self.o_t(x_and_h)
        
        next_c = prev_c * f_i + i_i * c_i
        next_h = torch.tanh(next_c) * o_i
        
        assert prev_h.shape == next_h.shape
        assert prev_c.shape == next_c.shape
        
        res_map = self.hidden_to_result(next_h)
        res = torch.zeros(holes_coordinates.shape[0], 3, device=DEVICE)
        for i in range(holes_coordinates.shape[0]):
            x = holes_coordinates[i, 0].item()
            y = holes_coordinates[i, 1].item()
            loc_map = (torch.zeros(2*self.out_rad - 1, 2*self.out_rad - 1, device=DEVICE) +
                       res_map[0, :, 
                              (x - self.out_rad + 1 + 3):(x + self.out_rad + 3),
                              (y - self.out_rad + 1 + 3):(y + self.out_rad + 3)])
#             print ("x and y", x, y)
#             print (res_map.shape)
            res[i, 0] = self.  oil_result(loc_map[0])
            res[i, 1] = self.water_result(loc_map[1])
            res[i, 2] = self.  gas_result(loc_map[2])
            
            
        
        return (next_c, next_h), res
        
    def init_state (self, batch_size, device=torch.device("cpu")):
        return (Variable(torch.ones(batch_size,
                                     self.hid_size,
                                     N_CELLS_HOR,
                                     N_CELLS_VER,
                                     device=device)),
                Variable(torch.ones(batch_size,
                                     self.hid_size,
                                     N_CELLS_HOR,
                                     N_CELLS_VER,
                                     device=device)))
               
    


# In[17]:


def save_model_state(model, state):
    torch.save(model.state_dict(), "model_state/state_dict")
    torch.save(state[0]          , "model_state/state_0")
    torch.save(state[1]          , "model_state/state_1")

def load_model_state():
    state=[]
    model = LSTMCell(N_FEATURES_INJECTION,
                     3,
                     embedding_size=16,
                     hidden_state_size=16,
                     output_lin_radius=4)
    model.load_state_dict(torch.load("model_state/state_dict"))
    state.append(torch.load("model_state/state_0"))
    state.append(torch.load("model_state/state_1"))
    return model, state
    


# In[18]:


def predict_values (RNN_cell,
                    hid_state,
                    device,
                    injection,
                    mining_coordinates):
    
    RNN_cell.eval().to(device)
    for elem in hid_state:
        elem.to(device)

    i = 0
        
    
    # hid_state = RNN_cell.init_state(batch_size=1, device=device)
        
    prediction_massive = []
    for t in range(injection.shape[0]):

        inputs      = injection[t].unsqueeze(0).to(device)
        coordinates = mining_coordinates[t].to(device)

#         for elem in hid_stat
        hid_state, prediction = RNN_cell.forward(inputs, hid_state, mining_coordinates[t])
        prediction_massive.append(prediction)
        
    return prediction_massive


# In[19]:


RNN_model, last_hid_state = load_model_state()
RNN_model.to(DEVICE)
for elem in last_hid_state:
    elem = elem.to(DEVICE)


# In[20]:


predict_massive = predict_values(RNN_model,
                                 last_hid_state,
                                 DEVICE,
                                 injection,
                                 mining_coordinates)


# In[21]:


def denorm_tensor(tensor):
    oil_mean, oil_std = 608.6271654299433, 465.1306866839296
    water_mean, water_std = 688.2310936395759, 777.8226797718818
    gas_mean, gas_std = 26261.32932862191, 20507.335485165913
    for j in range(tensor.shape[0]):
        tensor[j][0] = tensor[j][0]*oil_std + oil_mean 
        tensor[j][1] = tensor[j][1]*water_std + water_mean 
        tensor[j][2] = tensor[j][2]*gas_std + gas_mean 
    return tensor
        


# In[22]:


for elem in predict_massive:
    elem = denorm_tensor(elem)
#     print (elem.shape)


# In[23]:


mean = torch.zeros(predict_massive[0].shape)
for elem in predict_massive:
    mean += elem.to('cpu')
mean /= mean.shape[0]


# In[24]:


table = pd.DataFrame(columns=["x", "y", "oil", "water", "gas"],
                     index=range(mean.shape[0]))
for i in range (mean.shape[0]):
#     print (i)
    table.iloc[i, 0] = pump_data.iloc[i, 0]
    table.iloc[i, 1] = pump_data.iloc[i, 1]
    table.iloc[i, 2] = mean[i][0].item()
    table.iloc[i, 3] = mean[i][1].item()
    table.iloc[i, 4] = mean[i][2].item()
#     a = pump_data.iloc[i, 0]
#     table.concat([pump_data.iloc[i][0],
#                   pump_data.iloc[i][1],
#                   predict_massive[i][0],
#                   predict_massive[i][1],
#                   predict_massive[i][2]])
# table


# In[25]:


table.to_csv('result.csv', sep=';', header=False, index=False)

