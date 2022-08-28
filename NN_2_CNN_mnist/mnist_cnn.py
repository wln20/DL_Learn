# 导入需要的包
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm


# 设置超参数
BATCH_SIZE = 64 # 如果是笔记本电脑跑或者显卡显存较小，可以减小此值
LR = 0.1        # 学习率
MM = 0.9        # 随机梯度下降法中momentum参数
EPOCH = 10      # 训练轮数

# 设置pytorch使用的device，如果电脑有nvidia显卡，在安装cuda之后可以使用cuda，
# 如果没有，则使用默认的cpu即可，此句代码可以自动识别是否可以使用cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据集，数据集会存储在当前文件夹的'data/'子文件夹下，第一次执行
# 会下载数据集到此文件夹下
# 特别提醒，mnist是pytorch自带的数据集，可以通过此方法载入，如果需要导入自己
# 的数据，需要使用Dataset类实现，具体可参考给出的参考文献
train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.1307,), (0.3081,))
                ]),
    #download = True,            
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.1307,), (0.3081,))
                ]),
)

# 构建dataloader，pytorch输入神经网络的数据需要通过dataloader来实现
train_loader = torch.utils.data.DataLoader(
                    train_data, 
                    batch_size=BATCH_SIZE, 
                    shuffle=True, 
                    num_workers=1)  #num_workers是加载数据（batch）的线程数目

test_loader = torch.utils.data.DataLoader(
                    test_data, 
                    batch_size=BATCH_SIZE, 
                    shuffle=True, 
                    num_workers=1)

# 定义网络结构，简单的网络结构可以通过nn.Sequential来实现，复杂的
# 网络结构需要通过继承nn.Module来自定义网络类来实现，在此使用自定义
# 类的方法给出一个简单的卷积神经网络，包括两个卷积层和两个全连接层，

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # nn.Conv2d(in_channels,out_channels,kernel_size[,stride[,padding[,...]]])
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):     
                                # batch_size * channel_num * width * height
        x = self.conv1(x)       # 64*1*28*28 -> 64*32*26*26
        x = self.relu(x)
        x = self.max_pool(x)    # 64*32*26*26 -> 64*32*13*13
        x = self.conv2(x)       # 64*32*13*13 -> 64*64*11*11
        x = self.relu(x)
        x = self.max_pool(x)    # 64*64*11*11 -> 64*64*5*5
        x = self.dropout1(x)
        x = torch.flatten(x, 1) # 64*64*5*5 -> 64*1600
        x = self.fc1(x)         # 64*1600 -> 64*128
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)         # 64*128 -> 64*10
        return x
model = Net().to(device)

# 定义损失函数，分类问题采用交叉熵损失函数
loss_func = nn.CrossEntropyLoss()

# 定义优化方法，此处使用随机梯度下降法
optimizer_ft = optim.SGD(model.parameters(), lr=LR, momentum=MM)
# 定义每5个epoch，学习率变为之前的0.1
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

# 训练神经网络
# 这里将训练、测试都定义成函数，最后再调用。
# 每轮训练/测试都是把所有批的数据全输入一遍，也即每轮都要调用一次训练/测试函数
def train_model(model, criterion, optimizer, scheduler):

    # 训练模式
    model.train()

    running_loss = 0.0
    running_corrects = 0

    # 每次加载训练集的一批数据
    for inputs, labels in tqdm(train_loader):
        # 将数据加载到cpu/gpu上
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 输入该批数据，前向计算
        outputs = model(inputs)

        # 得到各图预测结果（label）
        _, preds = torch.max(outputs, 1)

        # 计算损失函数
        loss = criterion(outputs, labels)

        # 清除上一轮梯度
        optimizer.zero_grad()

        # 反向传播
        loss.backward()

        # 调整参数
        optimizer.step()

        # 统计该批数据的损失函数之和以及预测正确的样本数
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels)

    # 一个epoch结束，根据条件进行一次学习率调整
    scheduler.step()

    # 计算该轮中，全训练集的平均损失函数和平均预测正确率
    epoch_loss = running_loss / len(train_data)
    epoch_acc = running_corrects.double() /len(train_data)

    print('train Loss: {:.4f} Acc: {:.4f}'.format(
        epoch_loss, epoch_acc))

    # 返回训练好（调整参数完毕）的模型
    return model

# 测试神经网络
def test_model(model, criterion):

    # 测试模式
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    # torch.no_grad() 是一个上下文管理器，被该语句 wrap 起来的部分将不会track 梯度。
    with torch.no_grad():

        # 每次加载测试集的一批数据
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)
    
    epoch_loss = running_loss / len(test_data)
    epoch_acc = running_corrects.double() / len(test_data)

    print('test Loss: {:.4f} Acc: {:.4f}'.format(
        epoch_loss, epoch_acc))

    # 返回本轮测试的准确率
    return epoch_acc

# 正式开始训练和测试
if __name__ == "__main__":

    # 统计开始时间
    since = time.time()

    # 最大准确率
    best_acc = 0

    # 开始每轮训练
    for epoch in range(EPOCH):
        print('Epoch {}/{}'.format(epoch, EPOCH - 1))
        print('-' * 10)

        # 调用训练/测试函数
        model = train_model(model, loss_func, optimizer_ft, exp_lr_scheduler)
        epoch_acc = test_model(model, loss_func)

        # 根据本轮结果调整统计到的最大准确率
        best_acc = epoch_acc if epoch_acc > best_acc else best_acc
    
    # 统计总用时
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Acc: {:4f}'.format(best_acc))
