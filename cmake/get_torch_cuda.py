import torch

archs = [x[3:] for x in torch.cuda.get_arch_list() if int(x[3:])>=60]
print(";".join(archs), end='')
