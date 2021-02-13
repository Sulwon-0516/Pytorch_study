# -*- coding: utf-8 -*- 
# 본 신경망의 경우 LeNet이라고 한다..

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

class simple_net(nn.Module):
    # 이렇게 nn.Module을 안에 넣는 것은 상속을 받는다는 의미이다. 
    def __init__(self):
        #기존의 부모의 __init__의 선언 내용을 받아들이고 추가하겠다는 뜻.
        super(simple_net,self).__init__()
        
        # 1개의 이미지 입력을 받아서 6개의 channel로 출력되게 할 것
        # filter size : 3 x 3
        self.conv1 = nn.Conv2d(1,6,3)
        self.conv2 = nn.Conv2d(6,16,3)

        #Affine calculation으로 최종 채널에 맞게 FCL을 넣은 부분을 말한다.
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        # 2x2 로 Max-pooling을 가한다.
        # 이 부분에서 앞서 정의해둔 trainable한 값들을 이어준다.
        #여기서는 max_pool을 __init__에 정의하지 않고서 사용하지만, 뒤에선 다르다. 
        x = func.max_pool2d(func.relu(self.conv1(x)),(2,2))
        x = func.max_pool2d(func.relu(self.conv2(x)),2)

        #feature 개수를 반환해주는 함수 + view로 flatten하기
        x = x.view(-1, self.num_flat_features(x))
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def num_flat_features(self,x):
        # flatten을 해주는 함수
        # 첫 차원을 제외한 데이터를 가져옴. 첫 차원은 보통 batch 차원이기 때문.
        size = x.size()[1:] 
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# 신경망 정의
net = Net()
print(net)

# 파라미터 출력
params = list(net.parameters())
print(params)
print(len(params))
print(params[0].size())  # conv1의 .weightㅊ


# 정상 작동하는지 32x32의 임의의 데이터를 넣자. 
# batch와 color channel때문에 앞에 1이 두개 붙는다.
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

# 우선 grad를 영으로 비워주고, 무작위 grad에 대해서 역전파를 계산해본다.
net.zero_grad()
out.backward(torch.randn(1, 10))


#------------------------------------------------------------------#
print("--------------------------")
print("About Loss func \n")
# loss function을 정의해보자.

output = net(input)
target = torch.randn(10) #임의의 정답
target = target.view(1,-1)
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

print("back tracking")
print(loss.grad_fn)
print(loss.grad_fn.next_functions[0][0])
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])

#------------------------------------------------------------------#
print("--------------------------")
print("lets do back propagation\n")
# 역전파를 실행해보자. 

net.zero_grad()

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

#------------------------------------------------------------------#
print("--------------------------")
print("About SGD\n")
# 가중치를 갱신해보자.

learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)


# 위와 같이 SGD optimizer를 직접 만들어서 쓸 수도 있지만, 아래처럼 제공된 optimizer를 쓰는 것이 보편적이다. 
# Optimizer를 생성합니다.
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 학습 과정(training loop)에서는 다음과 같습니다:
optimizer.zero_grad()   # 변화도 버퍼를 0으로
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # 업데이트 진행