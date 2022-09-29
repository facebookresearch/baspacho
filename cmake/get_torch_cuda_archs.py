#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

archs = torch.cuda.get_arch_list()
archs = [arch for arch in archs if arch.startswith('sm_')]

# skip archs < 60 which do not support double atomicAdd, and the workaround
# cannot be compiled jointly with builtin atomicAdd of newer archs
archs = [arch[3:] for arch in archs if int(arch[3:]) >= 60]

print(";".join(archs), end='')
