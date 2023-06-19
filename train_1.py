from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt

from rsnet import rsnet34
from data_process import preprocess

path = r"C:\Users\limin\Desktop\1维DRSN\data_snr-5"
data_mark = "DE"
fs = 12000
win_tlen = 2048 / 12000
overlap_rate = (0 / 2048) * 100
random_seed = 1
batch_size=16
num_epochs=10
X, y = preprocess(path,
                  data_mark,
                  fs,
                  win_tlen,
                  overlap_rate,
                  random_seed
                  )
len_data=len(X)
print(len_data)
X=X.reshape(len_data,1,2048,1)
train_data = torch.from_numpy(X)
train_label = torch.from_numpy(y)
#train_data = type(torch.FloatTensor)
#train_label = type(torch.FloatTensor)
train_dataset = TensorDataset(train_data, train_label)

train_size = int(len(train_dataset) * 0.7)
test_size = len(train_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
test_data_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True,drop_last=True)

net = rsnet34()
net = net.cuda()

Loss_list = []
Accuracy_list = []
acc = []
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
loss_function = nn.CrossEntropyLoss()
for epoch in range(num_epochs):
    net.train()
    sum_loss = 0.0  # 损失数量
    correct = 0.0  # 准确数量
    total = 0.0  # 总共数量
    for i,(X,y) in enumerate(train_data_loader):
        length = len(train_data_loader)
        X = X.type(torch.cuda.FloatTensor)
        y = y.type(torch.cuda.LongTensor)
        #y = y.type(torch.cuda.FloatTensor)

        optimizer.zero_grad()
        outputs = net(X)
        loss = loss_function(outputs, y)
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += predicted.eq(y.data).cpu().sum()
        print('[epoch:%d, iter:%d/%d] Loss: %.03f | Acc: %.3f%% '
              % (epoch + 1, (i + 1), length, sum_loss / (i + 1), 100. * correct / total))
    Loss_list.append(sum_loss / (len(train_data_loader)))
    Accuracy_list.append(correct / total)

    print("Waiting Test!")
    with torch.no_grad():  # 没有求导
        correct = 0
        total = 0
        for test_i,(test_X,test_y) in enumerate(test_data_loader):
            net.eval()  # 运用net.eval()时，由于网络已经训练完毕，参数都是固定的，因此每个min-batch的均值和方差都是不变的，因此直接运用所有batch的均值和方差。
            X = test_X.type(torch.cuda.FloatTensor)
            y = test_y.type(torch.cuda.LongTensor)
            outputs = net(X)
            # 取得分最高的那个类 (outputs.data的索引号)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            if test_i == 100:
                break
        print('测试分类准确率为：{}%'.format(round(100 * correct / total, 3)))
        acc.append( 100. * correct / total)

x1 = range(0, num_epochs)
x2 = range(0, num_epochs)
x3 = range(0, num_epochs)
y1 = Accuracy_list
y2 = Loss_list
y3 = acc
plt.subplot(3, 1, 1)
plt.plot(x1, y1, 'o-')
plt.title('Train accuracy vs. epoches')
plt.ylabel('Train accuracy')
plt.subplot(3, 1, 2)
plt.plot(x2, y2, '.-')
plt.title('Train loss vs. epoches')
plt.ylabel('Train loss')
plt.subplot(3,1,3)
plt.plot(x3, y3, 'o-')
plt.title('Test accuracy vs. epoches')
plt.ylabel('Test accuracy')
plt.show()
#print(Accuracy_list)
print("Training Finished, TotalEPOCH=%d" % num_epochs)

