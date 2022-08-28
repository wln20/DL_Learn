#========================================================
#             Media and Cognition
#             Homework 1 Machine Learning Basics
#             classification.py - linear classification
#             Student ID:
#             Name:
#             Tsinghua University
#             (C) Copyright 2022
#========================================================

# ==== Part 0: import libs
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pylab import *
import sys
import torch.nn.functional as F



# ==== Part 1: definition of dataset
class MyDataset(Dataset):
    def __init__(self, file_path):
        '''
        :param file_path: the path of the npy file containing the features and labels of data
        The data in the npy file is an array of shape (N, 3), where the first two elements along the axis-1 are the features and the last element is the label of the data.
        '''
        # TODO: load all items of the dataset stored in file_path
        # you may need np.load() function
        super().__init__()
        self.data = np.load(file_path)
        self.feat = self.data[:,:2].astype(np.float32) # float32
        self.label = self.data[:,2].astype(np.long)
    def __len__(self):
        '''
        :return length: the number of items in the dataset
        '''
        # TODO: get the number of items in the dataset
        return self.data.shape[0]

    def __getitem__(self, index):
        '''
        :param index: the index of current item
        :return feature: an array of shape (2, ) and type np.float32
        :return label: an array of shape (1, ) and type np.long
        '''
        # TODO: get the feature and label of the current item
        # pay attention to the type of the outputs
        feat = self.feat[index]
        label = self.label[index]
        return feat,label


# ==== Part 2: network structure
# -- linear classifier
class Model(nn.Module):
    def __init__(self, input_size, output_size):
        '''
        :param input_size: dimension of input features
        :param output_size: number of classes, 1 for classification task of two class categories
        '''
        # TODO: initialization of the linear classifier, the model should include a linear layer and a Sigmoid layer
        # remember to initialize the father class and pay attention to the dimension of the model
        super().__init__()
        # self.weightT = nn.Parameter(torch.randn(output_size,input_size).T,requires_grad=True)
        # self.bias = nn.Parameter(torch.randn(output_size),requires_grad=True)
        
        self.linear = nn.Linear(input_size,output_size)
        #self.BatchNorm1 = nn.BatchNorm1d(output_size)
        self.act = nn.Sigmoid()


    def forward(self, x):
        '''
        :param x: input features of shape (batch_size, 2)
        :return pred: output of model of shape (batch_size, )
        '''
        # TODO: forward the model
        # pay attention that pred should be shape of (batch_size, ) rather than (batch_size, 1)
        
        # out = self.act(torch.matmul(x,self.weightT)+self.bias)   #matmul为矩阵乘法
        x = self.linear(x)
        #x = self.BatchNorm1(x)
        out = self.act(x)
        return out.view(-1,)


# ==== Part 3: define binary cross entropy loss for classification task

def bce_loss(pred, label):
    '''
    Binary cross entropy loss function
    ------------------------------
    :param pred: predictions with size [batch_size, *], * refers to the dimension of data
    :param label: labels with size [batch_size, *]
    :return: loss value, divided by the number of elements in the output
    '''
    # TODO: calculate the mean of losses for the samples in the batch
    
    
    #loss_layer = nn.CrossEntropyLoss()
    #pred = torch.tensor(pred,dtype=torch.float32,requires_grad=True)
    #label = torch.tensor(label,dtype=torch.int64)
    #pred = torch.stack([pred,(1-pred)])
    #pred = pred.T
    loss_layer = nn.BCELoss()
    loss = loss_layer(pred,label)
    #pred = pred-(1e-4)
    #loss = - torch.mean(label*torch.log(pred)+(1-label)*torch.log(1-pred))
    return loss



# ==== Part 4: training and validation
def train_val(train_file_path='data/character_classification/train_feat.npy',
              val_file_path='data/character_classification/val_feat.npy',
              n_epochs=20, batch_size=8, lr=1e-3, momentum=0.9, valInterval=5):
    '''
    The main training procedure
    ----------------------------
    :param train_file_path: file path of the training data
    :param val_file_path: file path of the validation data
    :param n_epochs: number of training epochs
    :param batch_size: batch size of training and validation
    :param lr: learning rate
    :param momentum: momentum of the stochastic gradient descent optimizer
    :param valInterval: the frequency of validation, e.g., if valInterval = 5, then do validation after each 5 training epochs
    :param device: 'cpu' or 'cuda', we can use 'cpu' for our homework if GPU with cuda support is not available
    '''

    # 输入训练数据和验证数据的.npy文件路径，利用自定义的dataset构建训练集和验证集
    traindata = MyDataset(train_file_path)
    valdata = MyDataset(val_file_path)

    # 构建加载训练集和验证集的Dataloader
    trainloader = DataLoader(traindata,batch_size=batch_size,shuffle=True)
    valloader = DataLoader(valdata,batch_size=batch_size,shuffle=False)

    # 将模型实例化，输入为2维（2个feature）
    model = Model(input_size=2,output_size=1)

    # 将模型加载到cpu or gpu上
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 将优化器实例化，使用随机梯度下降SGD
    optimizer = optim.SGD(model.parameters(),lr,momentum)

    # 保存每轮训练得到的总平均loss
    losses = []

    # 开始遍历各轮
    for epoch in range(n_epochs):

        # ===== training stage =====
        # 将模型设置为训练模式
        model.train()

        # 累积每批的平均loss之和，最后用于求平均
        total_loss = 0.0

        # 遍历训练集，step为批数，feat为(batch_size,2)形状的矩阵，label为 (batch_size,)的行向量
        for step,(feat,label) in enumerate(trainloader):    

            # 转换数据类型
            feat = feat.to(device).float()
            label = label.to(device).float()
            
            # 输入模型得到预测
            out = model(feat)

            # 计算损失函数（该批数据损失函数的均值）
            loss = bce_loss(out,label)

            # 反向传播前要清除上一轮的梯度
            optimizer.zero_grad()

            # 反向传播
            loss.backward()

            # 根据梯度调整参数
            optimizer.step()

            # 将该批的平均loss加入total_loss
            total_loss += loss.item()
        # 遍历train_loader结束

        # 计算本轮得到的总平均loss，并加入losses列表
        avg_loss = total_loss/len(trainloader)
        losses.append(avg_loss)
        print('Epoch {:02d}: loss = {:.3f}'.format(epoch+1,avg_loss))


        # ===== validation stage =====
        # 每valInterval轮（这里是每五轮）输入验证集数据进行一次验证，统计准确率（之前训练时则未统计准确率）
        if (epoch + 1) % valInterval == 0:

            # 将模型设置为测试验证模式
            model.eval()

            # 统计正确率
            n_correct = 0
            n_items = 0

            # 使用with torch.no_grad(): 可在测试验证时使用，不计算梯度
            with torch.no_grad():
                # 遍历验证集
                for step,(feat,label) in enumerate(valloader):
                    feat = feat.to(device)
                    label = label.to(device)
                    out = model(feat)
                    prediction = torch.round(out)       # 四舍五入来最终判定预测输出label是0还是1
                    n_correct += (prediction == label).sum()
                    n_items += feat.shape[0]

            # 统计验证集的正确率
            acc = n_correct/n_items
            print('Epoch {:02d}:, validation accuracy = {:.1f}%'.format(epoch+1,100*acc))

            # 保存模型到.pth文件（相当于每valInterval轮就保存一次模型）
            model_save_path = 'saved_models/model_epoch{:02d}.pth'.format(epoch + 1)
            torch.save(model.state_dict(),model_save_path)
            print('Model saved in {}\n'.format(model_save_path))

        

    # draw the loss curve
    plot_loss(losses)


# ==== Part 5: test
def test(model_path, test_file_path='data/character_classification/test_feat.npy', batch_size=8):
    '''
    Test procedure
    ---------------
    :param model_path: path of the saved model
    :param im_dir: path to directory with images
    :param test_file_path: file with test image paths and labels
    :param norm_size: image normalization size, (height, width)
    :param batch_size: test batch size
    :param device: 'cpu' or 'cuda'
    '''

    # initialize the model
    model = Model(input_size=2, output_size=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 构建好模型
    model = model.to(device)
    
    # 加载模型参数
    model.load_state_dict(torch.load(model_path, map_location=device))
    print('[Info] Load model from {}'.format(model_path))

    # call a function to enter the evaluation mode
    model.eval()

    # test loader
    testloader = DataLoader(MyDataset(test_file_path), batch_size=batch_size, shuffle=False)

    # run the test process
    n_correct = 0.
    n_ims = 0.
    with torch.no_grad():  # we do not need to compute gradients during the test stage

        for feats, labels in testloader:
            feats, labels = feats.to(device), labels.type(torch.float).to(device)
            out = model(feats)
            predictions = torch.round(out)
            n_correct += torch.sum(predictions == labels)
            n_ims += feats.size(0)

    print('[Info] Test accuracy = {:.1f}%'.format(100 * n_correct / n_ims))


# ==== Part 6: visualization

# plot the loss curve
def plot_loss(losses):
    '''
    :param losses: list of losses for each epoch
    :return:
    '''

    plt.figure()

    # draw loss
    plt.plot(losses)

    # set labels
    plt.xlabel('training epoch')
    plt.ylabel('loss')

    # show the plots
    plt.show()

# plot the classification result
def visual(model_path, file_path):
    '''
    Visualization: show feature distribution and the decision boundary
    ---------------------------------------------------------------
    :param model_path: path of the saved model
    :param im_dir: path to directory with images
    :param file_path: file with image paths and labels
    :param norm_size: image normalization size, (height, width)
    '''

    # construct data loader
    loader = DataLoader(MyDataset(file_path), batch_size=1, shuffle=False)

    # construct model
    model = Model(input_size=2, output_size=1)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    chars_real = []  # save features of real character images
    backgrounds_real = []  # save features of real background images
    chars_pred = []  # save features of predicted character images
    backgrounds_pred = []  # save features of predicted background images

    with torch.no_grad():

        for feat, label in loader:
            # run the model, get features and outputs
            feat = feat.detach()
            out = model(feat).detach()

            if label[0] == 1:  # real character images
                chars_real.append(feat.view(1, -1))
            else:  # real background images
                backgrounds_real.append(feat.view(1, -1))

            if out[0] > 0.5:  # predicted character images
                chars_pred.append(feat.view(1, -1))
            else:  # predicted background images
                backgrounds_pred.append(feat.view(1, -1))

    # convert list of tensors to an entire tensor
    chars_real = torch.cat(chars_real, 0)
    backgrounds_real = torch.cat(backgrounds_real, 0)
    chars_pred = torch.cat(chars_pred, 0)
    backgrounds_pred = torch.cat(backgrounds_pred, 0)

    # get weights and bias of the linear layer
    input_a = torch.tensor([1.0, 0.0]).float().view(1, 2)
    input_b = torch.tensor([0.0, 1.0]).float().view(1, 2)
    input_zeros = torch.tensor([0.0, 0.0]).float().view(1, 2)
    out_a = model(input_a)
    out_a = torch.log((out_a / (1.0 - out_a + 1e-7)) + 1e-7)
    out_b = model(input_b)
    out_b = torch.log((out_b / (1.0 - out_b + 1e-7)) + 1e-7)
    out_zeros = model(input_zeros)
    out_zeros = torch.log((out_zeros / (1.0 - out_zeros + 1e-7)) + 1e-7)
    weights = [[(out_a - out_zeros).item(),(out_b - out_zeros).item()]]
    bias = out_zeros.item()


    # compute slope and intercept of the dividing line
    a = - weights[0][0] / weights[0][1]
    b = - bias / weights[0][1]

    # draw features on a 2D surface
    f, (ax1, ax2) = plt.subplots(2, 1)
    subplots_adjust(hspace=0.3)
    ax1.scatter(chars_real[:, 0], chars_real[:, 1], marker='^', c='w', edgecolors='c', s=15)
    ax1.scatter(backgrounds_real[:, 0], backgrounds_real[:, 1], marker='o', c='w', edgecolors='m', s=15)

    ax1.set_xlabel('feature 1')
    ax1.set_ylabel('feature 2')
    ax1.legend(['real characters', 'real backgrounds'])

    ax2.scatter(chars_pred[:, 0], chars_pred[:, 1], marker='^', c='w', edgecolors='c', s=15)
    ax2.scatter(backgrounds_pred[:, 0], backgrounds_pred[:, 1], marker='o', c='w', edgecolors='m', s=15)

    # draw the dividing line
    x_min, x_max = ax2.get_xlim()
    y_min = x_min * a + b
    y_max = x_max * a + b
    ax2.plot([x_min, x_max], [y_min, y_max], 'r', linewidth=0.8)

    ax2.set_xlabel('feature 1')
    ax2.set_ylabel('feature 2')
    ax2.legend(['decision boundary', 'predicted characters', 'predicted backgrounds'])

    # show the plots
    plt.show()


# ==== Part 7: run
if __name__ == '__main__':
    # set random seed for reproducibility
    seed = 3
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    args = sys.argv

    if len(args) < 2 or args[1] not in ['train', 'test', 'visual']:
        print('Usage: $python classification.py [mode]')
        print('       mode should be one of [train, test, visual]')
        print('Example: python classification.py train')
        raise AssertionError

    mode = args[1]

    # -- run the code for training and validation
    if mode == 'train':
        train_val(train_file_path='data/character_classification/train_feat.npy',
                  val_file_path='data/character_classification/val_feat.npy', n_epochs=20, batch_size=8,
                  lr=0.01, momentum=0.9, valInterval=5)

    # -- test the saved model
    elif mode == 'test':
        test(model_path='saved_models/model_epoch20.pth',
             test_file_path='data/character_classification/train_feat.npy',
             batch_size=8)

    # -- visualization
    else:
        visual(model_path='saved_models/model_epoch20.pth',
               file_path='data/character_classification/test_feat.npy')