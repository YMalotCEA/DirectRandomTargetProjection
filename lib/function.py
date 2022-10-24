# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------------

Copyright (C) 2019 Université catholique de Louvain (UCLouvain), Belgium.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

------------------------------------------------------------------------------

 "function.py" - Functional definition of the TrainingHook class (module.py).
 
 Project: DRTP - Direct Random Target Projection

 Authors:  C. Frenkel and M. Lefebvre, Université catholique de Louvain (UCLouvain), 09/2019

 Cite/paper: C. Frenkel, M. Lefebvre and D. Bol, "Learning without feedback:
             Fixed random learning signals allow for feedforward training of deep neural networks,"
             Frontiers in Neuroscience, vol. 15, no. 629892, 2021. doi: 10.3389/fnins.2021.629892

------------------------------------------------------------------------------
"""


import torch
import torch.nn.functional as F
from torch.autograd import Function
from numpy import prod

def generalized_mm(input, matrix):
    assert input.shape[-1] == matrix.shape[0]
    common_dim = input.shape[-1]
    output_shape = input.shape[:-1] + matrix.shape[1:]
    return input.view(-1, common_dim).mm(matrix.view(common_dim, -1)).view(output_shape)


class HookFunction(Function):
    @staticmethod
    def forward(ctx, input, labels, y, fixed_fb_weights, train_mode):
        if train_mode in ["DFA", "sDFA", "DRTP"]:
            ctx.save_for_backward(input, labels, y, fixed_fb_weights)
        ctx.in1 = train_mode
        return input

    @staticmethod
    def backward(ctx, grad_output):
        train_mode = ctx.in1
        
        if train_mode == "BP":
            return grad_output, None, None, None, None
        elif train_mode == "shallow":
            grad_output.data.zero_()
            return grad_output, None, None, None, None
        
        input, labels, y, fixed_fb_weights = ctx.saved_variables
        if train_mode == "DFA":
            #TODO: modify (y - labels), which is true only when using the MSE
            feedback = y-labels
        elif train_mode == "sDFA":
            feedback = (y-labels).sign()
        elif train_mode == "DRTP":
            feedback = labels
        else:
            raise NameError("=== ERROR: training mode " + str(train_mode) + " not supported")
        
        grad_output_est = generalized_mm(feedback, fixed_fb_weights)
        return grad_output_est, None, None, None, None

trainingHook = HookFunction.apply
