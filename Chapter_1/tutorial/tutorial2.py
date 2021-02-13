# -*- coding: utf-8 -*- 

#it's about the autograd

import torch

x = torch.ones(2,2, requires_grad = True)
print(x)
# 기본 설정은 False임을 생각하자

y=x+2
print(y)
print(y.grad_fn)
print(y.grad)
print("-------------------")
z=y*y*3
out = z.mean()

print(x,out)
print("-------------------")
# requires_grad_에 대한 정보 + 여러 연산이 한번에 가해질 떄의 grad_fn의 값.
a = torch.randn(2,2)
a = ((a*3)/(a-1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a*a).sum()
print(b.grad_fn)
print("-------------------")

# 역전파를 실행해보자.
out.backward(torch.tensor(1.))

print(x.grad)
print(y.grad)
print("-------------------")
print("-------------------")
#1024배가 될때까지 진행될 것
#여기서부턴 x,y가 새롭게 정의된 케이스
x = torch.randn(3, requires_grad=True)

y = x * 2
i=0
while y.data.norm() < 1000:
    y = y * 2
    i = i+1
print(y)
print("count : ",i)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)
print("-------------------")

# no_grad 블락의 사용법
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)
print("-------------------")

# detach()의 사용법. 
# eq()로 확인을 해보면 전부 같은 결과를 가지고 있음을 볼 수 있다. 
print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())