
import torch

iscuda = torch.cuda.is_available()
device = "cuda" if iscuda else "cpu"
print(device)

t1 = torch.rand(2, 100000000, dtype=torch.float32, device=device)
t2 = torch.rand(2, 100000000, dtype=torch.float32, device=device)

t3 = t1 + t2

t3 = t3.sort()

print(t3)

t4 = torch.empty()
t4 = torch.zeros()
t4 = torch.ones()



