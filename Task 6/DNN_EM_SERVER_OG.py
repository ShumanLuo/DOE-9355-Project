
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import os
import time
from sklearn.model_selection import train_test_split


def reset_random_seeds():
    os.environ['PYTHONHASHSEED']=str(1)
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(0)
reset_random_seeds()

class modDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).float()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        a=1
        return x, y
    def __len__(self):
        return len(self.data)

class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.linear1 = nn.Linear(in_features=x_train_datasize, out_features=neuron_number)
        self.linear2 = nn.Linear(in_features=neuron_number, out_features=neuron_number)
        self.linear3 = nn.Linear(in_features=neuron_number, out_features=neuron_number)
        self.linear31 = nn.Linear(in_features=neuron_number, out_features=neuron_number)
        self.linear4 = nn.Linear(in_features=neuron_number, out_features=y_train_datasize)
        # self.linearn = nn.Linear(in_features=20, out_features=14)

    def forward(self, input):  # Input is a 1D tensor
        y = self.linear1(input)
        y = F.relu(self.linear2(y.clone()))
        y = F.relu(self.linear3(y.clone()))
        y = F.relu(self.linear31(y.clone()))
        y = F.relu(self.linear4(y.clone()))
        y = F.relu(y.clone())
        # y = F.relu(y)
        return y
def remove_numbers(numbers):
    multiples_of_670 = [i for i in range(0, max(numbers), 670) if i != 0]
    new_list = []
    for number in numbers:
        if any(abs(number-multiple) <= 10 for multiple in multiples_of_670):
            continue
        new_list.append(number)
    return new_list

testing_loss_list = list()
timer_list = list()

# for neuron in range(10,1000,5):
start = time.time()
neuron_number = 100
# adding the gausian white noise is standard
noise_mean = 0
noise_std = 1
samples = np.random.normal(noise_mean, noise_std, size= 1) # sample code for gaussian white noise

# using 200bus data event 1
train_data = pd.read_csv("Sample.csv")
# Vm_data = pd.read_csv
# train_data_2 =  pd.read_csv("C:\\Users\\zhihao's home\\Downloads\\testing.csv")
Dataset1 = train_data.values
# y_train = train_data.iloc[[10],[1]].values

bus_number = 200
PMU_for_training = 10 # number of PMU is selected while other converted to AMI data
PMU_selection_len = 10 # utilize 10 datapoint before to 10 datapoint after
Smart_mater_resolution = 30 # 15 --> one sample per 15min

# smart_meter_index = list(range(15, Dataset1.shape[0], Smart_mater_resolution))
smart_meter_index = list(range(PMU_selection_len, Dataset1.shape[0], Smart_mater_resolution))
all_time_step_index  = list(range(PMU_selection_len, Dataset1.shape[0]-PMU_selection_len, 1))

interpolate_remaining = [i for i in all_time_step_index if i not in smart_meter_index] # remove all the true dataset

smart_meter_index = remove_numbers(smart_meter_index) # remove the index that is within the correlation
all_time_step_index = remove_numbers(all_time_step_index)

if Dataset1.shape[0]-smart_meter_index[-1] < PMU_selection_len:
    smart_meter_index = smart_meter_index[0:-2] # make sure the last one have enough lenght

# randomly split columns into training and test sets
PMU_cols, Smart_meter_cols = train_test_split(
    train_data.columns,
    test_size=bus_number-PMU_for_training,
    random_state=42
)

# create separate DataFrames for training and test sets
Selected_PMU_array = train_data[PMU_cols].values
Selected_Smart_meter_array = train_data[Smart_meter_cols].values

sc = MinMaxScaler()
sct = MinMaxScaler()
X_train = list()
y_train = list()
X_test = list()
Y_test = list()
for i in smart_meter_index:
    X_train_sub = Selected_PMU_array[i-PMU_selection_len:i+PMU_selection_len,:]
    # X_train_sub = sc.fit_transform(X_train_sub.reshape(-1,1))
    # X_train_sub = X_train_sub.reshape(-1,1)
    # X_train_sub = X_train_sub.astype(np.float)
    y_train_sub = Selected_Smart_meter_array[i, :]
    # y_train_sub = sct.fit_transform(y_train_sub.reshape(-1,1))
    # y_train_sub = y_train_sub.reshape(-1,1)
    # y_train_sub = y_train_sub.astype(np.float)
    X_train.append(X_train_sub)
    y_train.append(y_train_sub)
for i in interpolate_remaining:
    X_test_sub = Selected_PMU_array[i-PMU_selection_len:i+PMU_selection_len,:]
    # X_test_sub = sc.fit_transform(X_test_sub.reshape(-1,1))
    # X_test_sub = X_test_sub.reshape(-1,1)
    # X_test_sub = X_test_sub.astype(np.float)
    y_test_sub = Selected_Smart_meter_array[i, :]
    # y_test_sub = sct.fit_transform(y_test_sub.reshape(-1,1))
    # y_test_sub = y_test_sub.reshape(-1,1)
    # y_test_sub = y_test_sub.astype(np.float)
    X_test.append(X_test_sub)
    Y_test.append(y_test_sub)

# y_train =sct.fit_transform(y_train.reshape(-1,1))
x_train_datasize = 2*PMU_selection_len*PMU_for_training
y_train_datasize = Dataset1.shape[1]- PMU_for_training

x_train_flat = []
x_test_flat = []
y_train_flat = []
y_test_flat = []

# loop through each 2D ndarray in the list and flatten it into a 1D ndarray
for arr in X_train:
    flat_arr = arr.reshape(-1)  # reshape
    x_train_flat.append(flat_arr)

for arr in y_train:
    flat_arr = arr.reshape(-1)  # reshape
    y_train_flat.append(flat_arr)

for arr in X_test:
    flat_arr = arr.reshape(-1)  # reshape
    x_test_flat.append(flat_arr)

for arr in Y_test:
    flat_arr = arr.reshape(-1)  # reshape
    y_test_flat.append(flat_arr)

# stack the flattened arrays vertically into a single 2D ndarray
X_train = np.vstack(x_train_flat)
y_train = np.vstack(y_train_flat)
X_test = np.vstack(x_test_flat)
Y_test = np.vstack(y_test_flat)

X_test_total = torch.from_numpy(X_test)
y_test_total = torch.from_numpy(Y_test)

X_train_total = torch.from_numpy(X_train).float()
y_train_total = torch.from_numpy(y_train).float()
# total_dataset = FuckMyDataset(xtrain,ytrain)

# loader = torch.utils.data.DataLoader(dataset=total_dataset,batch_size=50,shuffle=True,num_workers=0)



model = network()

learning_rate = 1E-6
torch.autograd.set_detect_anomaly(True)
l = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr =learning_rate )
iteration_step = 100
y_test_predict_tensor = torch.empty(1,y_train_datasize) # initialize the predit

training_loss_list = list()

predit_data_list_sum = list() # this is the list for storing all the prediction in every iteration step

for iter in range(iteration_step):
    # adam
    optimizer.zero_grad()
    if iter == 0:
        new_X_train = X_train_total.float().clone()
        new_Y_train = y_train_total.float().clone()
    if iter > 0:
        new_X_train = torch.cat((X_train_total.float().clone(),X_test_total.float().clone()))
        new_Y_train = torch.cat((y_train_total.float().clone(),y_test_predict_tensor.float().clone()))
        print("New Z is formed")
        #load_saved_model
        state = torch.load("F:/EM_Model/model"+str(iter)+".pth")
        model = network()
        model.load_state_dict(state['model'])
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
        optimizer.load_state_dict(state['optimizer'])
        for parameter in model.parameters():
            parameter.requires_grad = True
    new_dataset = torch.utils.data.TensorDataset(new_X_train,new_Y_train.clone())
    loader = torch.utils.data.DataLoader(dataset=new_dataset,batch_size=200,shuffle=True,num_workers=0)
    print("new model and dataset is loaded")

    for epoch in range(500):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = l(outputs, labels)


            loss.backward(retain_graph=True)

            optimizer.step()
            #clear out the gradients from the last step loss.backward()
        if (epoch+1) % 400 == 0:
            print('epoch {}, loss {}'.format(epoch, loss.item()))
            testing_loss_MSE = nn.MSELoss()
            y_test_predict_tensor = model(X_test_total.float())
            testing_loss = testing_loss_MSE(y_test_predict_tensor, y_test_total).item()
            testing_loss_list.append(testing_loss)
            testing_loss_bb = np.array(testing_loss_list)
            predict_interpolated_data = y_test_predict_tensor.detach().numpy()
            truth_interpolated_data = y_test_total.detach().numpy()
            Original_SCADA_point = y_train
            predict_interpolated_data_with_original_point = np.insert(predict_interpolated_data, smart_meter_index,
                                                                      Original_SCADA_point, axis=0)
            predit_data_list_sum.append(predict_interpolated_data)
            truth_interpolated_data_with_original_point = np.insert(truth_interpolated_data, smart_meter_index,
                                                                    Original_SCADA_point, axis=0)

            # Draw the graph with different markers for the subset
            plt.plot(predict_interpolated_data_with_original_point[:, 2], '-',color='blue')
            plt.plot(truth_interpolated_data_with_original_point[:, 2], color='green')
            # Define the markers for the graph
            plt.scatter(smart_meter_index, Original_SCADA_point[:, 2], s=100, marker='o', facecolors='none',
                        edgecolors='red')

            # Show the graph
            plt.show()

            # calculate the percentage error
            epsilon = 0
            y_test_total_no_zero = y_test_total + epsilon
            y_test_predict_tensor_no_zero = y_test_predict_tensor + epsilon
            pct_error = torch.mean(
                torch.abs((y_test_predict_tensor_no_zero - y_test_total_no_zero) / y_test_total_no_zero)) * 100
            print(pct_error.item())
            a=1

    # get the updated predict matrix for the Z
    print("Iteration finished")

    # testing_loss_MSE = nn.MSELoss()
    # y_test_predict_tensor = model(X_test_total.float())
    # testing_loss = testing_loss_MSE(y_test_predict_tensor,y_test_total).item()
    # testing_loss_list.append(testing_loss)
    # testing_loss_bb = np.array(testing_loss_list)
    # predict_interpolated_data = y_test_predict_tensor.detach().numpy()
    # truth_interpolated_data = y_test_total.detach().numpy()
    # Original_SCADA_point = y_train
    # predict_interpolated_data_with_original_point = np.insert(predict_interpolated_data, smart_meter_index,
    #                                                           Original_SCADA_point, axis=0)
    # truth_interpolated_data_with_original_point = np.insert(truth_interpolated_data, smart_meter_index,
    #                                                         Original_SCADA_point, axis=0)
    #
    # # Draw the graph with different markers for the subset
    # plt.plot(predict_interpolated_data_with_original_point[:, 2], color='blue')
    # plt.plot(truth_interpolated_data_with_original_point[:, 2], color='green')
    # # Define the markers for the graph
    # plt.scatter(smart_meter_index,Original_SCADA_point[:, 2], s=100, marker='o', facecolors='none', edgecolors='red')
    #
    # # Show the graph
    # plt.show()

    #calculate the percentage error
    epsilon = 0
    y_test_total_no_zero = y_test_total + epsilon
    y_test_predict_tensor_no_zero = y_test_predict_tensor + epsilon
    pct_error = torch.mean(
        torch.abs((y_test_predict_tensor_no_zero - y_test_total_no_zero) / y_test_total_no_zero)) * 100
    print(pct_error.item())
    a=1
    # save the model and parameter for the next iteration
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    model_save_path = "F:/EM_Model/model"+str(iter+1)+".pth"
    print("DNN_model_saved")
    torch.save(state, model_save_path)
    model.zero_grad()
    a =1

testing_loss_MSE = nn.MSELoss()
y_test_predict_tensor = model(X_test_total.float())
testing_loss = testing_loss_MSE(y_test_predict_tensor,y_test_total).item()
testing_loss_list.append(testing_loss)
testing_loss_bb = np.array(testing_loss_list)
print(testing_loss_bb[0])







end = time.time()
timer_list.append(end-start)
print(end-start)
# pd.DataFrame(final_predict_data).to_csv("final_predict_data_DNN.csv")
# final_test_data = y_test_total.detach().numpy()
# pd.DataFrame(final_test_data).to_csv("final_test_data.csv")
a=1
# this is the remaining iteration begin
# iteration_number = 1
#
# for iter in range(iteration_number):
#
#     # xtrain = np.array(X_train).reshape(len(smart_meter_index),x_train_datasize)
#     # ytrain = np.array(y_train).reshape(len(smart_meter_index),y_train_datasize)
#     # total_dataset = FuckMyDataset(xtrain,ytrain)
#     # trainloader = torch.utils.data.DataLoader(X_train, batch_size=4,
#     #                                           shuffle=True, num_workers=0)
#     loader = torch.utils.data.DataLoader(dataset=new_dataset,batch_size=50,shuffle=True,num_workers=0)
#     # network.reset_parameters()
#
#     y_test_predict_tensor = model2(X_test_total.float())        # print('epoch {}, loss {}'.format(epoch, loss.item()))
# a = 1
