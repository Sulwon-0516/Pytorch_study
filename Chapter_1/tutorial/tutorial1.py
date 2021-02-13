from __future__ import print_function
import torch

x= torch.empty(5,3)
print(x)

#from __future__ import print_function으로 print를 불러오지 않아도, py내장된거로 작동하는군.
#empty는 rand라기보단, trash값에 가깝다.

x=torch.rand(5,3)
print(x)

#특징적인 것은 rand의 형태를 바꿀수는 없다. (그러니까 분포를)
#다만 device 설정을 처리 device를 CPU/GPU를 바꿀수 있고, 추가로 autoGrad를 추적할지 여부도 명시할 수 있다.

x=torch.zeros(5,3,dtype=torch.long)
print(x)


x=torch.tensor([[5.5,3]])
print(x)
print(x.size())

# 하나라도 double형이면, 전부 double형이 된다.
# 가장 바깥 괄호부터 차원이 메겨지는 것임.

x=x.new_ones(5,3, dtype=torch.double)
print(x)


x=torch.randn_like(x,dtype=torch.float)
print(x)

# 이전의 x size는 입력으로 넣어주지 않으면, new를 써도 영향을 안주는 듯?
#d_type을 overload한다?

print(x.size())
# tuple형태이기에, tuple로 연산이 가능하다고 함.



y=torch.rand(5,3)
print(x+y)

z=torch.rand(3,5)
#print(x+z.transpose())
#print(x+z)
#위의 사례에서 볼 수 있듯이, 크기가 맞지않으면 error가 발생한다.


print(torch.add(x,y))
print(x.add_(y))
print(x.div_(y))

result = torch.empty(5,3)
torch.add(x,y,out=result)
print(result)
#위와 같이 함수형으로도 선언을 할 수 있다는 것이 특징적임. (out을)
#_을 붙이면 +=과 같이 작동을 한다는 것도 특징적임. (in-place 방식)


print(x[:,2])
# MATLAB이나 np스러운 indexing을 사용해도 된다.

# tesnor의 size와 shape을 변경할때는 view라는 함수를 사용한다.
# -1로 적혀있으면, 다른 차원을 기준으로 계산하고, 나머지를 의미한다.

x=torch.randn(4,4)
y=x.view(16)
z=x.view(-1,8)

print(x.size(),y.size(),z.size());

#q=x.view(-1,5)
#위와같이 적합하지 않은 형태로 바꿀경우 오류를 발생시킨다.

# tensor에 하나의 값만 존재하면, .item()을 써서 숫자값을 얻을 수 있음
# 만약 1개의 값만 존재하는 것이 아니면, 에러를 발생시킴.

x=torch.randn(1)

print(y)
print(x)
print(x.item())



# 일단 NP array로 바꾸는 것 부터 보자.
a=torch.randn(5)
print(a)

b=a.numpy()
print(b)
# 근데 NP를 import안했는데도 작동하네

a.add_(1)
# 이렇게 해주면, 자동으로 ones*1을 곱해주는구나.
# 얘네들은 변수가 연동형으로 작동을 하네..b=a.numpy()는 사실상 포인터의 정의라 봐도 무방하다.

print(a)
print(b)

# 그러면 NP arrray를 Torch로 바꿔보자.
import numpy as np
a=np.ones(5)
b=torch.from_numpy(a)
np.add(a,1,out=a)
print(a)
print(b)
a=2*a
print(a)
print(b)

# from_numpy()의 경우 연동성이 있지는 않다.



# CUDA가 사용가능한 환경에서, GPU <> CPU간 데이터 이동을 보자.

if torch.cuda.is_available():
    device = torch.device("cuda")
    y=torch.ones_like(x,device=device)
    x=x.to(device)
    z=x+y
    print(z)
    print(z.to("cpu",torch.double))