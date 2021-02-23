# -*- coding: utf-8 -*- 

import os
import torch
import torch.nn as nn
import torch.nn.functional as func
import LeNet_classifier
import cifar10_data_loader as cf

# 배치 크기는 40으로 설정되어 있다. 
LEARNING_RATE = 0.001
MOMENTUM = 0.9
EPOCH = 5

# GPU상의 학습을 위한 밑밥
device = torch.device("cuda:0" if torch.cuda.is_available() else " cpu")
print(device)

#make the network & get the data
net = LeNet_classifier.LeNet()
trainloader,testloader = cf.load_cifar10()

'''
net.to(device)
'''

#define the optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr = LEARNING_RATE, momentum = MOMENTUM)

#train the Neural Network

for epoch in range(EPOCH):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # input, labels 을 입력 받자.
        inputs, labels = data

        # inputs, labels = data[0].to(device), data[1].to(device)

        # gradient의 매개변수를 0으로 초기화해주자. 
        optimizer.zero_grad()

        # forward + backward + optim
        outputs = net(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        # 통계 데이터를 출력한다. 
        running_loss += loss.item()
        # 2000번 미니 배치마다 출력을 한다. 
        if i % 2000 == 1999: 
            print('[%d, %5d] loss: %.3f' %
            (epoch+1, i+1, running_loss/2000))
            running_loss = 0.0

print('Finished Training')

# 저장할 폴더 이름이 없을 경우 이렇게 수행한다.
FOLDER = './trained_data'
PATH = 'Lenet.pth'
if not os.path.exists(FOLDER):
    os.makedirs(FOLDER)
torch.save(net.state_dict(), os.path.join(FOLDER,PATH))



