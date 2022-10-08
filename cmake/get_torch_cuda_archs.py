#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

archs = torch.cuda.get_arch_list()
archs = [arch[3:] for arch in archs if arch.startswith('sm_')]

print(";".join(archs), end='')
