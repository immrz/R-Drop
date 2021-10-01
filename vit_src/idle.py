import torch
import time

a = torch.rand(1000000, 10).to("cuda:0")
b = torch.rand(1000000, 10).to("cuda:1")
c = torch.rand(1000000, 10).to("cuda:2")
d = torch.rand(1000000, 10).to("cuda:3")

e = torch.rand(1000000, 10).to("cuda:4")
f = torch.rand(1000000, 10).to("cuda:5")
g = torch.rand(1000000, 10).to("cuda:6")
h = torch.rand(1000000, 10).to("cuda:7")

while True:
    a = a * a
    b = b * b
    c = c * c
    d = d * d
    e = e * e
    f = f * f
    g = g * g
    h = h * h
    time.sleep(0.01)
