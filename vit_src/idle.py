import torch
import time

a = torch.rand(1000000, 10).to("cuda:0")
b = torch.rand(1000000, 10).to("cuda:1")
c = torch.rand(1000000, 10).to("cuda:2")
d = torch.rand(1000000, 10).to("cuda:3")

while True:
    a = a * a
    b = b * b
    c = c * c
    d = d * d
    time.sleep(0.01)
