# -*- coding: utf-8 -*- 

import cifar10_data_loader as cf
import torch
import torchvision
import matplotlib.pyplot as plt 
import numpy as np

#이미지를 출력해주는 함수
def imshow(img):
    #이미 normalize되어 있으니, 풀력을 위해서 다시 변환해준다. 
    img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()


def main():
    trainloader,testloader = cf.load_cifar10()
    
    #무작위로 데이터를 가져오기. 
    dataiter = iter(trainloader)
    images , labels = dataiter.next()

    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % cf.classes[labels[j]] for j in range(4)))