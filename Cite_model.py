import os
import pandas as pd
import numpy as np
import tables
from sklearn.decomposition import TruncatedSVD
import torch
from torch import nn
#from d2l import torch as d2l
from torch.utils.data import DataLoader,TensorDataset
import torch.nn.init as init
import random


DATA_DIR = "/home/jdhan_pkuhpc/profiles/lijianzhe/gpfs1/open-problems-multimodal/"
FP_CELL_METADATA = os.path.join(DATA_DIR,"metadata.csv")

FP_CITE_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_cite_inputs.h5")
FP_CITE_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_cite_targets.h5")
FP_CITE_TEST_INPUTS = os.path.join(DATA_DIR,"test_cite_inputs.h5")

PROCESSED_DATA_DIR = "/home/jdhan_pkuhpc/profiles/lijianzhe/gpfs1/processed/"

FP_CITE_TRAIN_DATA=os.path.join(PROCESSED_DATA_DIR,"processed_cite_train_data.h5")
FP_CITE_TEST_DATA=os.path.join(PROCESSED_DATA_DIR,"processed_cite_test_data.h5")
FP_CITE_TARGET_DATA=os.path.join(PROCESSED_DATA_DIR,"processed_cite_target_data.h5")

MODEL_SAVE_DIR="/home/jdhan_pkuhpc/profiles/lijianzhe/gpfs1/model/"

MODEL_SAVEPATH=os.path.join(MODEL_SAVE_DIR,"net.params")
RESULT_SAVEPATH=os.path.join(MODEL_SAVE_DIR,"reuslts.xlsx")


def save_model(results_df, train_ls, valid_ls, train_acc, valid_acc, i, num_epochs):
    fold_results = pd.DataFrame({
        'Epoch': list(range(1, num_epochs + 1)),
        'Fold': [i + 1] * num_epochs,
        'Train Loss': train_ls,
        'Validation Loss': valid_ls,
        'Train Accuracy': train_acc,
        'Validation Accuracy': valid_acc
    })
    return pd.concat([results_df, fold_results], ignore_index=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(0)

def show_data(data1,data2=np.arange(1)):
    print(data1)
    print(data1.shape)
    print(data2)
    print(data2.shape)

def load_input_data(file_path):
    with tables.File(file_path, mode='r') as hdf_file:
        # 读取数据
        data_array = hdf_file.get_node('/df')[:]
        # 读取列名
        column_names_array = hdf_file.get_node('/columns')[:]
        column_names = [name.decode('utf-8') for name in column_names_array]
        # 读取行标签
        row_labels_array = hdf_file.get_node('/row_labels')[:]
        row_labels = [label.decode('utf-8') for label in row_labels_array]
        # 创建 DataFrame
        return pd.DataFrame(data_array, index=row_labels, columns=column_names)
    
def load_train_data(file_path):
    with tables.File(file_path, mode='r') as hdf_file:
        reduced_data = hdf_file.get_node('/reduced_data')[:]
        original_columns = hdf_file.get_node('/original_columns')[:].astype(str)
        original_row_labels = hdf_file.get_node('/original_row_labels')[:].astype(str)
        scaler_mean = hdf_file.get_node('/scaler_mean')[:]
        scaler_scale = hdf_file.get_node('/scaler_scale')[:]
        column_medians = hdf_file.get_node('/column_medians')[:]
        tsvd_components = hdf_file.get_node('/tsvd_components')[:]


    # 创建 DataFrame
    df_reduced = pd.DataFrame(reduced_data, index=original_row_labels, columns=original_columns)

    return df_reduced,scaler_mean,scaler_scale,column_medians,tsvd_components

def label_gender(donor_id):
    if donor_id == 13176:
        return 'FEMALE'
    else:
        return 'MALE'
    
def pred_from_output(output,scaler_mean,scaler_scale,column_medians,tsvd_components):
    
    reconstructed_data = output @ tsvd_components

    # 加回中位数
    reconstructed_data += column_medians

    # 逆标准化
    original_space_data = (reconstructed_data * scaler_scale) + scaler_mean
    return original_space_data
train_data = load_input_data(FP_CITE_TRAIN_DATA)

#test_data = load_input_data(FP_CITE_TEST_DATA)

df_meta = pd.read_csv(FP_CELL_METADATA, index_col='cell_id')

#print(train_data.iloc[:,0])
#print(train_data.columns)

# 将metadata中的cell_type与train数据合并
final_cite_train_data = train_data.merge(df_meta[['cell_type', 'donor', 'day']], left_index=True, right_index=True, how='left')
#final_cite_test_data = test_data.merge(df_meta[['cell_type', 'donor', 'day']], left_index=True, right_index=True, how='left')
#print(final_cite_train_data.columns)


final_cite_train_data['gender'] = final_cite_train_data['donor'].apply(label_gender)
#final_cite_test_data['gender'] = final_cite_test_data['donor'].apply(label_gender)

#print(final_cite_train_data)
#print(final_cite_test_data.dtypes)

#all_features = pd.concat((final_cite_train_data.iloc[:, 1:], final_cite_test_data.iloc[:, 1:]))
#print(all_features)

#print(final_cite_train_data)

#train_data_features = final_cite_train_data.drop(columns=['cell_id'])
#test_data_features = final_cite_test_data.drop(columns=['cell_id'])

final_cite_train_data['donor'] = final_cite_train_data['donor'].astype(str)
final_cite_train_data['day'] = final_cite_train_data['day'].astype(str)

#final_cite_test_data['donor'] = final_cite_test_data['donor'].astype(str)
#final_cite_test_data['day'] = final_cite_test_data['day'].astype(str)


final_cite_train_data = pd.get_dummies(final_cite_train_data, dummy_na=True)
#final_cite_test_data = pd.get_dummies(final_cite_test_data, dummy_na=True)

#show_data(final_cite_train_data,final_cite_test_data)
#print(final_cite_train_data.columns)

#print(final_cite_train_data.values)

"""
for col in final_cite_train_data.columns:
    if(final_cite_train_data[col].dtype!='float32'):
        print(f"Column '{col}' has type {final_cite_train_data[col].dtype}")
    #print(col,'aaaaaaaaaaaaa')

# 
for col in final_cite_test_data.columns:
    if final_cite_test_data[col].dtype == 'object':
        print(f"Column '{col}' has type {final_cite_test_data[col].dtype}")
"""

"""
train_nan_columns = final_cite_train_data.columns[final_cite_train_data.isnull().any()].tolist()
print("Train data columns with NaN values:", train_nan_columns)

test_nan_columns = final_cite_test_data.columns[final_cite_test_data.isnull().any()].tolist()
print("Test data columns with NaN values:", test_nan_columns)
"""

# 强制转换为 float32 类型
final_cite_train_data = final_cite_train_data.astype(np.float32)
#final_cite_test_data = final_cite_test_data.astype(np.float32)


train_features = torch.tensor(final_cite_train_data.values, dtype=torch.float32)
#test_features = torch.tensor(final_cite_test_data.values, dtype=torch.float32)

target_data,scaler_mean,scaler_scale,column_medians,tsvd_components = load_train_data(FP_CITE_TARGET_DATA)
#print(target_data)
#print(target_data)
#print(target_data.dtypes)
#print(target_data.columns)

target_features=torch.tensor(target_data.values, dtype=torch.float32)
#show_data(train_features,test_features)
#show_data(target_features)

target_value=pd.read_hdf(FP_CITE_TRAIN_TARGETS)
target_true_value=torch.tensor(target_value.values, dtype=torch.float32)

print(train_features)
print(target_true_value)
#print(pred_from_output(target_data,scaler_mean,scaler_scale,column_medians,tsvd_components))
#print(target_value)

#------------------------------preprocessing finished--------------------------------

loss = nn.MSELoss()
in_features = train_features.shape[1]

 


def calculate_pearson(y_true, y_pred):
    corr_sum = 0
    for i in range(len(y_true)):
        # 检查预测值是否全部相同
        if np.all(y_pred[i] == y_pred[i][0]):
            corr_sum += -1.0
        else:
            corr_sum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
    
    return corr_sum / len(y_true)

def evaluate_loss(net, features, labels):
    with torch.no_grad():
        # 将数据转换为张量
        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        y_pred = net(features)
        return loss(y_pred, labels).item()

def evaluate_pearson(net, features, labels):
    net.eval()  # 确保模型处于评估模式
    with torch.no_grad():
        features = features.to('cuda')  # 将特征数据移到 GPU
        labels = labels.to('cuda')  # 将标签数据移到 GPU
        pred_features = net(features).detach()  # 在 GPU 上计算预测
        pred_features = pred_from_output(pred_features.cpu().numpy(), scaler_mean, scaler_scale, column_medians, tsvd_components)  # 将结果移到 CPU 并转换为 NumPy 数组
        labels = labels.cpu().numpy()  # 将标签移到 CPU 并转换为 NumPy 数组

    return calculate_pearson(pred_features, labels)



def train(net, train_features, train_labels, test_features, test_labels,target_train,target_valid,
          num_epochs, learning_rate, weight_decay, batch_size):
    
    train_ls, test_ls = [], []
    train_acc, test_acc = [], []
    train_dataset = TensorDataset(train_features, train_labels)
    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    for epoch in range(num_epochs):
        net.train()
        for X, y in train_iter:
            X, y = X.cuda(), y.cuda()
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
            print("okkkkkkkk")
            train_ls.append(evaluate_loss(net, train_features, train_labels))
            train_acc.append(evaluate_pearson(net,train_features, target_train))
            if test_labels is not None:
                net.eval()
                test_ls.append(evaluate_loss(net, test_features, test_labels))
                test_acc.append(evaluate_pearson(net,test_features, target_valid))
        print(f'epoach{epoch + 1}，训练loss{float(train_ls[-1]):f}, '
              f'验证loss{float(test_ls[-1]):f}')
        print(f'epoach{epoch + 1}，训练acc{float(train_acc[-1]):f}, '
              f'验证acc{float(test_acc[-1]):f}')
    return train_ls, test_ls,train_acc, test_acc


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.layer_norm1 = nn.LayerNorm(in_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features, in_features)
        self.layer_norm2 = nn.LayerNorm(in_features)

    def forward(self, x):
        identity = x
        out = self.relu(self.layer_norm1(self.fc1(x)))
        out = self.layer_norm2(self.fc2(out))
        out += identity
        out = self.relu(out)
        return out


class MyNetwork(nn.Module):
    def __init__(self, in_features):
        super(MyNetwork, self).__init__()
        # 现有网络部分
        self.fc1 = nn.Linear(in_features, 16384)
        self.relu = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(16384)
        
        self.fc2 = nn.Linear(16384, 9128)
        self.layer_norm2 = nn.LayerNorm(9128)
        
        self.fc3 = nn.Linear(9128, 4096)
        self.layer_norm3 = nn.LayerNorm(4096)
        
        self.fc4 = nn.Linear(4096, 2048)
        self.layer_norm4 = nn.LayerNorm(2048)
        
        # 新增的5个残差块
        self.res_blocks1 = nn.Sequential(
            ResidualBlock(2048),
            ResidualBlock(2048),
        )

        self.fc5 = nn.Linear(2048,1024)
        self.layer_norm5 = nn.LayerNorm(1024)
        self.fc6 = nn.Linear(1024,512)
        self.layer_norm6 = nn.LayerNorm(512)

        self.res_blocks2 = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512),
        )

        self.fc7 = nn.Linear(512,256)
        self.layer_norm7 = nn.LayerNorm(256)

        self.fc8 = nn.Linear(256,128)
        


    def forward(self, x):
        # 现有网络部分
        x = self.relu(self.layer_norm1(self.fc1(x)))
        x = self.relu(self.layer_norm2(self.fc2(x)))
        x = self.relu(self.layer_norm3(self.fc3(x)))
        x = self.relu(self.layer_norm4(self.fc4(x)))
        
        # 增加的5个残差块
        x = self.res_blocks1(x)

        x = self.relu(self.layer_norm5(self.fc5(x)))
        x = self.relu(self.layer_norm6(self.fc6(x)))

        x = self.res_blocks2(x)
        x = self.relu(self.layer_norm7(self.fc7(x)))
        x = self.fc8(x)

        return x

def get_net():
    net = MyNetwork(in_features)
    net.to('cuda')
    net.apply(initialize_weights)  # 应用权重初始化
    return net
    
    
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

def get_k_fold_data(k, i, X, y,target_true_value):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part, target_part= X[idx, :], y[idx,:],target_true_value[idx,:]
        if j == i:
            X_valid, y_valid ,target_valid= X_part, y_part, target_part
        elif X_train is None:
            X_train, y_train, target_train= X_part, y_part, target_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
            target_train=torch.cat([target_train,target_part ], 0)
    
   
    return X_train.to('cuda'), y_train.to('cuda'), X_valid.to('cuda'), y_valid.to('cuda'), target_train.to('cuda'), target_valid.to('cuda')

def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size,target_true_value):
    train_l_sum, valid_l_sum = 0, 0
    train_acc_sum, valid_acc_sum = 0, 0

    results_df = pd.DataFrame()

    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train,target_true_value)
        net = get_net()
        net.to('cuda')
        train_ls, valid_ls ,train_acc, valid_acc= train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        train_acc_sum += train_acc[-1]
        valid_acc_sum += valid_acc[-1]
        
        results_df = save_model(results_df, train_ls, valid_ls, train_acc, valid_acc, i, num_epochs)




        if i == 0:
            torch.save(net.state_dict(), MODEL_SAVEPATH)
        print(f'折{i + 1}，训练loss{float(train_ls[-1]):f}, '
              f'验证loss{float(valid_ls[-1]):f}')
            
    return train_l_sum / k, valid_l_sum / k, train_acc_sum/k, valid_acc_sum/k,results_df

k=5
batch_size = 200
lr = 0.001
weight_decay = 1e-4
num_epochs = 60

train_l, valid_l,train_acc,valid_acc,results_df= k_fold(k, train_features, target_features, num_epochs, lr,weight_decay, batch_size,target_true_value)

results_df.to_excel(RESULT_SAVEPATH, index=False)

#print("train_loss: ",train_l)
#print("valid_loss: ",valid_l)
#print("train_acc: ",train_acc)
#print("valid_acc: ",valid_acc)


#print(pred_from_output(target_data,scaler_mean,scaler_scale,column_medians,tsvd_components))