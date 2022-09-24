import torch

archs = torch.cuda.get_arch_list()
archs = [arch for arch in archs if arch.startswith('sm_')]

# skip archs < 60 which do not support double atomicAdd, and the workaround
# cannot be compiled jointly with builtin atomicAdd of newer archs
archs = [arch[3:] for arch in archs if int(arch[3:]) >= 60]

print(";".join(archs), end='')
