import torch
import torch.nn as nn
# import math
# import numpy as np
import torch.nn.functional as F
from torch.autograd import Function
# from utils.options import args

# class BinarizeConv2d(nn.Conv2d):

#     def __init__(self, *kargs, **kwargs):
#         super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
#         self.alpha = nn.Parameter(torch.rand(self.weight.size(0), 1, 1), requires_grad=True)
#         self.register_buffer('tau', torch.tensor(1.))
#         self.delta = nn.Parameter(torch.tensor(2.), requires_grad=True)
#         self.rho = nn.Parameter(torch.tensor(0.3), requires_grad=False)
        

#     def forward(self, input):
#         a = input
#         w = self.weight

#         w0 = w - w.mean([1,2,3], keepdim=True)
#         w1 = w0 / (torch.sqrt(w0.var([1,2,3], keepdim=True) + 1e-5) / 2 / np.sqrt(2))
#         EW = torch.mean(torch.abs(w1))
#         Q_tau = (- EW * torch.log(2-2*self.tau)).detach().cpu().item()
#         w2 = torch.clamp(w1, -Q_tau, Q_tau)

#         if self.training:
#             a0 = a / torch.sqrt(a.var([1,2,3], keepdim=True) + 1e-5) 
#         else: 
#             a0 = a
        
#         #* binarize 
#         bw = BinaryQuantize().apply(w2)
        
#         # ba = BinaryQuantize_a().apply(a0)
#         ba = BinaryQuantize_a().apply(a0, self.delta, self.rho)
        
#         #* 1bit conv
#         output = F.conv2d(ba, bw, self.bias,
#                           self.stride, self.padding,
#                           self.dilation, self.groups)
#         #* scaling factor
#         output = output * self.alpha
#         return output

class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
        self.alpha = nn.Parameter(torch.rand(self.weight.size(0), 1, 1), requires_grad=True)
        self.register_buffer('tau', torch.tensor(1.))

        self.delta = nn.Parameter(torch.tensor(2.), requires_grad=True)
        self.rho = nn.Parameter(torch.tensor(0.3), requires_grad=False)


    def forward(self, input):
        a = input
        w = self.weight

        w0 = w - w.mean([1,2,3], keepdim=True)
        w1 = w0 / (torch.sqrt(w0.var([1,2,3], keepdim=True) + 1e-5))

        a0 = a
        
        #* binarize 
        bw = BinaryQuantize().apply(w1)

        # ba = BinaryQuantize_a().apply(a0)
        ba = BinaryQuantize_a().apply(a0, self.delta, self.rho)
        
        #* 1bit conv
        output = F.conv2d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        #* scaling factor
        output = output * self.alpha
        return output

class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

# class BinaryQuantize_a(Function):
#     @staticmethod
#     def forward(ctx, input):
#         ctx.save_for_backward(input)
#         out = torch.sign(input)
#         return out
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         input = ctx.saved_tensors[0]
#         grad_input = (2 - torch.abs(2*input))
#         grad_input = grad_input.clamp(min=0) * grad_output.clone()
#         return grad_input


class BinaryQuantize_a(Function):
    @staticmethod
    def forward(ctx, input, delta, rho):
        ctx.save_for_backward(input, delta, rho)
        out = (input > 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        delta = ctx.saved_tensors[1]
        rho = ctx.saved_tensors[2]
        z = (input <= delta ) * ( - rho * delta <= input).float()
        grad_input = z.div(delta)
        grad_input = grad_input * grad_output.clone()
        grad_delta = (grad_output * z * (- input)).div(delta).div(delta).sum()
        return grad_input, grad_delta, None



# class BinaryQuantize_a(Function):
#     @staticmethod
#     def forward(ctx, input):
#         ctx.save_for_backward(input)
#         out = (input > 0).float()
#         return out
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         input = ctx.saved_tensors[0]
#         grad_input = (2 - torch.abs(4*input))
#         grad_input = grad_input.clamp(min=0) * grad_output.clone()
#         return grad_input




class BinaryQuantize_a_ste(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = (torch.abs(input) <= 1).float()
        grad_input = grad_output.clone()
        return grad_input


class BinaryQuantize_a_quadric(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = (2 - torch.abs(2*input))
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input
    


#Straight Throught Estimator surrogate gradient 
class BinaryQuantize_a_s_ste(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = (input > 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = (torch.abs(input) <= 0.5).float()
        grad_input = grad_input * grad_output.clone()
        return grad_input    
    

#Esser et al. quadric surrogate function
class BinaryQuantize_a_s_quadric(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = (input > 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = (1 - torch.abs(2*input))
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input    


#Sigmoid Function
class BinaryQuantize_a_s_sigmoid(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = (input > 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = 5*torch.sigmoid(5*input) * (1 - torch.sigmoid(5*input))
        # grad_input = grad_input.clamp(min=0) * grad_output.clone()
        grad_input = grad_input * grad_output.clone()
        return grad_input        


# Zenke et al. SuperSpike

class BinaryQuantize_a_s_super(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = (input > 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = 1 / (2 * torch.abs(input) + 1) ** 2
        grad_input = grad_input * grad_output.clone()
        return grad_input  

# class BinaryQuantize_a(Function):
#     @staticmethod
#     def forward(ctx, input, theta, delta):
#         ctx.save_for_backward(input, theta, delta)
#         # out = torch.sign(input)
#         out = (input >= theta).float()
#         return out

#     @staticmethod
#     def backward(ctx, grad_output):
#         input = ctx.saved_tensors[0]
#         theta = ctx.saved_tensors[1]
#         delta = ctx.saved_tensors[2]
#         z = (input <= theta + delta) * (theta - 0.3 * delta <= input).float()
#         grad_input = z.div(delta)
#         grad_input = grad_input * grad_output.clone()
#         grad_theta = (-grad_output * z).div(delta).sum()
#         grad_delta = (grad_output * z * (theta - input)) .div(delta).div(delta).sum()
    
#         return grad_input, grad_theta, grad_delta


# Wang et al. Learnable STE
class BinaryQuantize_a_s_lste(Function):
    @staticmethod
    def forward(ctx, input, delta, rho):
        ctx.save_for_backward(input, delta, rho)
        out = (input > 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        delta = ctx.saved_tensors[1]
        rho = ctx.saved_tensors[2]
        z = (input <= delta ) * ( - rho * delta <= input).float()
        grad_input = z.div(delta)
        grad_input = grad_input * grad_output.clone()
        grad_delta = (grad_output * z * (- input)).div(delta).div(delta).sum()
        return grad_input, grad_delta, None

# class BinaryQuantize_a_s_lste(Function):
#     @staticmethod
#     def forward(ctx, input, delta, rho):
#         ctx.save_for_backward(input)
#         out = (input > 0).float()
#         return out

#     @staticmethod
#     def backward(ctx, grad_output):
#         input = ctx.saved_tensors[0]
#         z = (input <= 1 ) * ( - 0.3 <= input).float()
#         grad_input = z * grad_output.clone()
#         return grad_input, None, None

class BinaryQuantize_a_s_aisin(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = (input > 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]

        # grad_input = 1 / (1 + input) * (input > 0).float() + 5*torch.sigmoid(5*input) * (1 - torch.sigmoid(5*input))* (input<=0).float()
        # grad_input = 1 / (1 + input) * (input > 0).float() + 5*torch.sigmoid(5*input) * (1 - torch.sigmoid(5*input))* (input<=0).float()
        
        grad_input = (1 - 2 * input) * (input > 0).float() + (1 - 4 * input)* (input<=0).float()
        grad_input = grad_input.clamp(min=0)

        grad_input = grad_input * grad_output.clone()
        return grad_input


