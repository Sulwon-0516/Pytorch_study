# -*- coding: utf-8 -*- 

import os
import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision
import numpy as np

import LeNet_classifier
import cifar10_data_loader as cf


FOLDER = './trained_data'
PATH = 'Lenet.pth'

# 테스트를 위해서 데이터를 가져온다. 
trainloader, testloader = cf.load_cifar10()

net = LeNet_classifier.Lenet()
net.load_state_dict(torch.load(os.path.join(FOLDER,PATH)))

# test data에서 이미지와 레이블을 설정하자.
# test의 경우 배치가 4로 설정해두었다.
dataiter = iter(testloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % cf.classes[labels[j]] for j in range(4)))

# 이제 예측을 돌려보자.
outputs = net(images)
#앞에는 아마 최고 값을 반환 할 것임
_, predicted = torch.max(outputs, 1)
print('Predicted: ',''.join('%5s' % cf.classes[outputs[j]] for j in range(4)))


#이제 모든 test data에 대해서 어떻게 동작하는 지를 출력해보자. 
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        # 배치로 평가를 진행하기 때문에, 이렇게 sum을 통해서 합쳐줘야 한다.
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


print('------------------------------\n')
# 이제는 어떤 경우에 더 잘 학습하는 지를 확인해보는 방법이다. 

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


