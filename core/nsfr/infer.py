# import random

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from nsfr.utils.torch_utils import softor, weight_sum


# def init_identity_weights(X, device):
#     ones = torch.ones((X.size(0),), dtype=torch.float32) * 100
#     return torch.diag(ones).to(device)


# class InferModule(nn.Module):
#     """
#     A class of differentiable foward-chaining inference.
#     """

#     def __init__(self, I, m, infer_step, gamma=0.01, device=None, train=False):
#         """
#         In the constructor we instantiate two nn.Linear modules and assign them as
#         member variables.
#         """
#         super(InferModule, self).__init__()
#         self.I = I
#         self.infer_step = infer_step
#         # m is num of clauses
#         self.m = m
#         self.C = self.I.size(0)
#         self.G = self.I.size(1)
#         self.gamma = gamma
#         self.device = device
#         self.train_ = train
#         if not train:
#             self.W = self.init_identity_weights(device)
#         else:
#             # to learng the clause weights, initialize W as follows:
#             self.W = nn.Parameter(torch.Tensor(
#                 np.random.normal(size=(m, I.size(0)))).to(device))
#         # clause functions
#         self.cs = [ClauseFunction(i, I, gamma=gamma)
#                    for i in range(self.I.size(0))]

#         # assert m == self.C, "Invalid m and C: " + \
#         #     str(m) + ' and ' + str(self.C)

#     def init_identity_weights(self, device):
#         ones = torch.ones((self.C,), dtype=torch.float32) * 100
#         return torch.diag(ones).to(device)

#     def forward(self, x):
#         """
#         In the forward function we accept a Tensor of input data and we must return
#         a Tensor of output data. We can use Modules defined in the constructor as
#         well as arbitrary operators on Tensors.
#         """
#         R = x
#         for t in range(self.infer_step):
#             R = softor([R, self.r(R)], dim=1, gamma=self.gamma)
#         return R

#     # def r(self, x):
#     #     B = x.size(0)  # batch size
#     #     # apply each clause c_i and stack to a tensor C
#     #     # C * B * G
#     #     C = torch.stack([self.cs[i](x)
#     #                      for i in range(self.I.size(0))], 0)

#     #     # taking weighted sum using m weights and stack to a tensor H
#     #     # m * C
#     #     W_star = torch.softmax(self.W, 1)
#     #     # print(W_star)
#     #     # m * C * B * G
#     #     W_tild = W_star.unsqueeze(
#     #         dim=-1).unsqueeze(dim=-1).expand(self.m, self.C, B, self.G)
#     #     # m * C * B * G
#     #     C_tild = C.unsqueeze(dim=0).expand(self.m, self.C, B, self.G)
#     #     # m * B * G
#     #     H = torch.sum(W_tild * C_tild, dim=1)
#     #     # taking soft or to compose a logic program with m clauses
#     #     # B * G
#     #     R = softor(H, dim=0, gamma=self.gamma)
#     #     return R

#     def r(self, x):
#         B = x.size(0)

#         # C * B * G  — one ClauseFunction call per clause
#         C = torch.stack([self.cs[i](x) for i in range(self.I.size(0))], dim=0)

#         # m * C
#         W_star = torch.softmax(self.W, dim=1)

#         # FIX 3: replace (m,C,B,G) materialisation with einsum
#         # Original:
#         #   W_tild = W_star.unsqueeze(-1).unsqueeze(-1).expand(m, C, B, G)  # big alloc
#         #   C_tild = C.unsqueeze(0).expand(m, C, B, G)                      # big alloc
#         #   H = torch.sum(W_tild * C_tild, dim=1)                           # (m,B,G)
#         # einsum contracts the C dimension directly — (m,C,B,G) never exists in memory
#         H = torch.einsum('mc,cbg->mbg', W_star, C)  # (m, B, G)

#         # soft-or over m clause results → (B, G)
#         R = softor(H, dim=0, gamma=self.gamma)
#         return R

#     def get_params(self):
#         return self.W


# class ClauseBodyInferModule(nn.Module):
#     """
#     A class of differentiable foward-chaining inference.
#     """

#     def __init__(self, I, gamma=0.01, device=None, train=False):
#         """
#         In the constructor we instantiate two nn.Linear modules and assign them as
#         member variables.
#         """
#         super(ClauseBodyInferModule, self).__init__()
#         self.I = I
#         self.C = self.I.size(0)
#         self.G = self.I.size(1)
#         self.gamma = gamma
#         self.device = device
#         self.train_ = train

#         # clause functions
#         # self.cs_bs = [ClauseBodySumFunction(I[i], I, gamma=gamma)
#         #               for i in range(self.I.size(0))]
#         self.cs_bs = [ClauseBodySumFunction(i, I, gamma=gamma)
#                       for i in range(self.I.size(0))]
#         self.cs = [ClauseFunction(i, I, gamma=gamma)
#                    for i in range(self.I.size(0))]

#     def init_identity_weights(self, device):
#         ones = torch.ones((self.C,), dtype=torch.float32) * 100
#         return torch.diag(ones).to(device)

#     def forward(self, x):
#         """
#         In the forward function we accept a Tensor of input data and we must return
#         a Tensor of output data. We can use Modules defined in the constructor as
#         well as arbitrary operators on Tensors.
#         """
#         # return self.r(x)
#         return self.r_cb(x)

#     def r(self, x):
#         B = x.size(0)  # batch size
#         # apply each clause c_i and stack to a tensor C
#         # C * B * G
#         C = torch.stack([self.cs[i](x)
#                          for i in range(self.I.size(0))], 0)
#         return C

#     def r_cb(self, x):
#         # x: C * B * G
#         B = x.size(1)  # batch size
#         # apply each clause c_i and stack to a tensor C
#         # C * B * G
#         # infer from i-th valuation tensor using i-th clause
#         # C = torch.stack([self.cs_bs[i](x[i])
#         #                  for i in range(self.I.size(0))], 0)
#         # print(self.cs_bs[0](x))
#         # print(self.cs_bs[0](x[0]))
#         C = torch.stack([self.cs_bs[i](x)
#                          for i in range(self.I.size(0))], 0)
#         return C

#     def get_params(self):
#         return self.W


# class ClauseInferModule(nn.Module):
#     def __init__(self, I, infer_step, gamma=0.01, device=None, train=False, m=1, I_bk=None):
#         """
#         Infer module using each clause.
#         The result is not amalgamated in terms of clauses.
#         """
#         super(ClauseInferModule, self).__init__()
#         self.I = I
#         self.I_bk = I_bk
#         self.infer_step = infer_step
#         self.m = m
#         self.C = self.I.size(0)
#         self.G = self.I.size(1)
#         self.gamma = gamma
#         self.device = device
#         self.train_ = train
#         if not train:
#             self.W = init_identity_weights(I, device)
#         else:
#             # to learng the clause weights, initialize W as follows:
#             self.W = nn.Parameter(torch.Tensor(
#                 np.random.normal(size=(m, I.size(0)))).to(device))
#         # clause functions
#         self.cs = [ClauseFunction(I[i], I, gamma=gamma)
#                    for i in range(self.I.size(0))]

#         self.cs_bs = [ClauseBodySumFunction(I[i], I, gamma=gamma)
#                       for i in range(self.I.size(0))]

#         if not self.I_bk is None:
#             self.cs_bk = [ClauseFunction(I_bk[i], I, gamma=gamma)
#                           for i in range(self.I_bk.size(0))]

#         if not I_bk is None:
#             self.W_bk = init_identity_weights(I_bk, device)

#         assert m == self.C, "Invalid m and C: " + \
#                             str(m) + ' and ' + str(self.C)

#     def forward(self, x):
#         """
#         In the forward function we accept a Tensor of input data and we must return
#         a Tensor of output data. We can use Modules defined in the constructor as
#         well as arbitrary operators on Tensors.
#         """
#         return self.r_cb(x)

#     def r_cb(self, x):
#         # x: C * B * G
#         B = x.size(1)  # batch size
#         # apply each clause c_i and stack to a tensor C
#         # C * B * G
#         # infer from i-th valuation tensor using i-th clause
#         C = torch.stack([self.cs_bs[i](x[i])
#                          for i in range(self.I.size(0))], 0)
#         return C


# # class ClauseFunction(nn.Module):
# #     """
# #     A class of the clause function.
# #     """

# #     def __init__(self, i, I, gamma=0.01):
# #         super(ClauseFunction, self).__init__()
# #         self.i = i  # clause index
# #         self.I = I  # index tensor C * S * G, S is the number of possible substituions
# #         self.L = I.size(-1)  # number of body atoms
# #         self.S = I.size(-2)  # max number of possible substitutions
# #         self.gamma = gamma

# #     def forward(self, x):
# #         batch_size = x.size(0)  # batch size
# #         # B * G
# #         V = x
# #         # G * S * b
# #         I_i = self.I[self.i, :, :, :]

# #         # B * G -> B * G * S * L
# #         V_tild = V.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.S, self.L)
# #         # G * S * L -> B * G * S * L
# #         I_i_tild = I_i.repeat(batch_size, 1, 1, 1)

# #         # B * G
# #         C = softor(torch.prod(torch.gather(V_tild, 1, I_i_tild), 3),
# #                    dim=2, gamma=self.gamma)
# #         return C

# # class ClauseFunction(nn.Module):
# #     def __init__(self, i, I, gamma=0.01):
# #         super(ClauseFunction, self).__init__()
# #         self.i     = i
# #         self.I     = I
# #         self.L     = I.size(-1)
# #         self.S     = I.size(-2)
# #         self.gamma = gamma

# #     def forward(self, x):
# #         batch_size = x.size(0)
# #         V   = x                         # (B, G)
# #         I_i = self.I[self.i]            # (G, S, L)

# #         # expand() = zero-copy view, unlike repeat() which allocates new memory
# #         V_tild   = V.unsqueeze(-1).unsqueeze(-1).expand(batch_size, V.size(1), self.S, self.L)
# #         I_i_tild = I_i.unsqueeze(0).expand(batch_size, -1, -1, -1)

# #         # torch.gather backward is scatter_add — fast, unlike advanced indexing backward
# #         C = softor(torch.prod(torch.gather(V_tild, 1, I_i_tild), dim=3), dim=2, gamma=self.gamma)
# #         return C


# class ClauseFunction(nn.Module):
#     """
#     Evaluates clause i over a batch of valuations.

#     FIX 1+2: replace .repeat() with .expand() (zero-copy views) so the
#     (B, G, S, L) tensor is never physically allocated.  torch.gather is
#     kept (not replaced with advanced indexing) because its backward is a
#     fast scatter_add, whereas advanced-indexing backward is much slower.
#     """

#     def __init__(self, i, I, gamma=0.01):
#         super(ClauseFunction, self).__init__()
#         self.i     = i
#         self.I     = I
#         self.L     = I.size(-1)   # number of body literals
#         self.S     = I.size(-2)   # max substitutions
#         self.gamma = gamma

#     def forward(self, x):
#         batch_size = x.size(0)
#         V   = x                            # (B, G)
#         I_i = self.I[self.i]              # (G, S, L)

#         # expand() = zero-copy view, no memory allocated for either tilde tensor
#         V_tild   = V.unsqueeze(-1).unsqueeze(-1).expand(batch_size, V.size(1), self.S, self.L)
#         I_i_tild = I_i.unsqueeze(0).expand(batch_size, -1, -1, -1)

#         # gather backward = scatter_add (fast); advanced-indexing backward is slow
#         C = softor(
#             torch.prod(torch.gather(V_tild, 1, I_i_tild), dim=3),
#             dim=2,
#             gamma=self.gamma,
#         )
#         return C
# # class ClauseBodySumFunction(nn.Module):
# #     """
# #     A class of the clause function.
# #     """

# #     def __init__(self, i, I, gamma=0.01):
# #         super(ClauseBodySumFunction, self).__init__()
# #         self.i = i  # clause index
# #         self.I = I  # index tensor C * S * G, S is the number of possible substituions
# #         self.L = I.size(-1)  # number of body atoms
# #         self.S = I.size(-2)  # max number of possible substitutions
# #         self.gamma = gamma

# #     def forward(self, x):
# #         batch_size = x.size(0)  # batch size
# #         # B * G
# #         V = x
# #         # G * S * b
# #         I_i = self.I[self.i, :, :, :]

# #         # B * G -> B * G * S * L
# #         V_tild = V.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.S, self.L)
# #         # G * S * L -> B * G * S * L
# #         I_i_tild = I_i.repeat(batch_size, 1, 1, 1)

# #         # B * G
# #         C = torch.sum(torch.prod(torch.gather(V_tild, 1, I_i_tild), 3), dim=2)
# #         return C



# class ClauseBodySumFunction(nn.Module):
#     """
#     Same repeat→expand fix as ClauseFunction, but aggregates with sum
#     instead of softor.

#     FIX 1+2 (same as ClauseFunction): .repeat() → .expand() to avoid
#     allocating the (B, G, S, L) intermediate tensors.
#     """

#     def __init__(self, i, I, gamma=0.01):
#         super(ClauseBodySumFunction, self).__init__()
#         self.i = i
#         self.I = I
#         self.L = I.size(-1)
#         self.S = I.size(-2)
#         self.gamma = gamma

#     def forward(self, x):
#         batch_size = x.size(0)
#         V   = x                            # (B, G)
#         I_i = self.I[self.i]              # (G, S, L)

#         # FIX: expand instead of repeat — zero-copy, no allocation
#         V_tild   = V.unsqueeze(-1).unsqueeze(-1).expand(batch_size, V.size(1), self.S, self.L)
#         I_i_tild = I_i.unsqueeze(0).expand(batch_size, -1, -1, -1)

#         C = torch.sum(
#             torch.prod(torch.gather(V_tild, 1, I_i_tild), dim=3),
#             dim=2,
#         )
#         return C


import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nsfr.utils.torch_utils import softor, weight_sum


def init_identity_weights(X, device):
    ones = torch.ones((X.size(0),), dtype=torch.float32) * 100
    return torch.diag(ones).to(device)


class InferModule(nn.Module):
    """
    A class of differentiable forward-chaining inference.
    """

    def __init__(self, I, m, infer_step, gamma=0.01, device=None, train=False):
        super(InferModule, self).__init__()
        self.register_buffer('I', I)  # buffer: DataParallel moves to each GPU
        self.infer_step = infer_step
        self.m = m
        self.C = I.size(0)
        self.G = I.size(1)
        self.S = I.size(2)
        self.L = I.size(3)
        self.gamma = gamma
        self.device = device
        self.train_ = train
        if not train:
            self.W = self.init_identity_weights(device)
        # else:
        #     self.W = nn.Parameter(torch.Tensor(
        #         np.random.normal(size=(m, I.size(0)))).to(device))
        else:
            W_init = torch.zeros(m, I.size(0))
            for i in range(min(m, I.size(0))):
                W_init[i, i] = 3.0   # softmax(3.0) ≈ 0.95 — strong but not saturated
            self.W = nn.Parameter(W_init.to(device))


        # nn.ModuleList so DataParallel can see and replicate these submodules
        # to each GPU. Plain Python lists are invisible to DataParallel and
        # would cause device mismatch errors in multi-GPU training.
        self.cs = nn.ModuleList([ClauseFunction(i, I, gamma=gamma)
                                 for i in range(self.I.size(0))])

    def init_identity_weights(self, device):
        ones = torch.ones((self.C,), dtype=torch.float32) * 100
        return torch.diag(ones).to(device)

    def forward(self, x):
        R = x
        for t in range(self.infer_step):
            R = softor([R, self.r(R)], dim=1, gamma=self.gamma)
        return R

    def r(self, x):
        B = x.size(0)

        # Loop over clauses — each ClauseFunction processes (B, G, S, L)
        # independently and frees memory before the next call.
        C_stack = torch.stack([self.cs[i](x) for i in range(self.C)], dim=0)  # (C, B, G)

        # einsum avoids materializing the full (m, C, B, G) intermediate tensor
        W_star = torch.softmax(self.W, dim=1)                  # (m, C)
        H = torch.einsum('mc,cbg->mbg', W_star, C_stack)      # (m, B, G)

        R = softor(H, dim=0, gamma=self.gamma)
        return R

    def get_params(self):
        return self.W


class ClauseBodyInferModule(nn.Module):
    """
    A class of differentiable forward-chaining inference.
    """

    def __init__(self, I, gamma=0.01, device=None, train=False):
        super(ClauseBodyInferModule, self).__init__()
        self.register_buffer('I', I)  # buffer: DataParallel moves to each GPU
        self.C = I.size(0)
        self.G = I.size(1)
        self.gamma = gamma
        self.device = device
        self.train_ = train

        # nn.ModuleList for DataParallel compatibility
        self.cs_bs = nn.ModuleList([ClauseBodySumFunction(i, I, gamma=gamma)
                                    for i in range(self.I.size(0))])
        self.cs = nn.ModuleList([ClauseFunction(i, I, gamma=gamma)
                                 for i in range(self.I.size(0))])

    def init_identity_weights(self, device):
        ones = torch.ones((self.C,), dtype=torch.float32) * 100
        return torch.diag(ones).to(device)

    def forward(self, x):
        return self.r_cb(x)

    def r(self, x):
        C = torch.stack([self.cs[i](x)
                         for i in range(self.I.size(0))], dim=0)
        return C

    def r_cb(self, x):
        C = torch.stack([self.cs_bs[i](x)
                         for i in range(self.I.size(0))], dim=0)
        return C

    def get_params(self):
        return self.W


class ClauseInferModule(nn.Module):
    def __init__(self, I, infer_step, gamma=0.01, device=None, train=False, m=1, I_bk=None):
        super(ClauseInferModule, self).__init__()
        self.register_buffer('I', I)       # buffer: DataParallel moves to each GPU
        self.register_buffer('I_bk', I_bk if I_bk is not None else torch.zeros(1))
        self.has_I_bk = I_bk is not None
        self.infer_step = infer_step
        self.m = m
        self.C = I.size(0)
        self.G = I.size(1)
        self.gamma = gamma
        self.device = device
        self.train_ = train
        if not train:
            self.W = init_identity_weights(I, device)
        else:
            self.W = nn.Parameter(torch.Tensor(
                np.random.normal(size=(m, I.size(0)))).to(device))

        # nn.ModuleList for DataParallel compatibility
        self.cs = nn.ModuleList([ClauseFunction(i, I, gamma=gamma)
                                 for i in range(self.I.size(0))])
        self.cs_bs = nn.ModuleList([ClauseBodySumFunction(i, I, gamma=gamma)
                                    for i in range(self.I.size(0))])
        if self.has_I_bk:
            self.cs_bk = nn.ModuleList([ClauseFunction(i, I_bk, gamma=gamma)
                                        for i in range(I_bk.size(0))])
            self.W_bk = init_identity_weights(I_bk, device)

        assert m == self.C, "Invalid m and C: " + str(m) + ' and ' + str(self.C)

    def forward(self, x):
        return self.r_cb(x)

    def r_cb(self, x):
        C = torch.stack([self.cs_bs[i](x[i])
                         for i in range(self.I.size(0))], dim=0)
        return C


class ClauseFunction(nn.Module):
    """
    Evaluates clause i over a batch of valuations.
    expand() instead of repeat() = zero-copy views, no memory allocated.
    torch.gather backward = fast scatter_add.

    I_i is registered as a buffer so DataParallel automatically moves it
    to the correct GPU for each worker. Plain tensor attributes stay on
    GPU 0 and cause device mismatch errors in multi-GPU training.
    """

    def __init__(self, i, I, gamma=0.01):
        super(ClauseFunction, self).__init__()
        self.L     = I.size(-1)
        self.S     = I.size(-2)
        self.gamma = gamma
        # register_buffer: automatically moved to the correct GPU by DataParallel
        self.register_buffer('I_i', I[i])  # (G, S, L)

    def forward(self, x):
        batch_size = x.size(0)
        V = x  # (B, G)

        # expand = zero-copy views, no allocation for either tilde tensor
        V_tild   = V.unsqueeze(-1).unsqueeze(-1).expand(batch_size, V.size(1), self.S, self.L)
        I_i_tild = self.I_i.unsqueeze(0).expand(batch_size, -1, -1, -1)

        C = softor(
            torch.prod(torch.gather(V_tild, 1, I_i_tild), dim=3),
            dim=2,
            gamma=self.gamma,
        )
        return C


class ClauseBodySumFunction(nn.Module):
    """
    Same expand+gather fix as ClauseFunction, aggregates with sum instead of softor.
    I_i registered as buffer for DataParallel GPU placement.
    """

    def __init__(self, i, I, gamma=0.01):
        super(ClauseBodySumFunction, self).__init__()
        self.L = I.size(-1)
        self.S = I.size(-2)
        self.gamma = gamma
        # register_buffer: automatically moved to the correct GPU by DataParallel
        self.register_buffer('I_i', I[i])  # (G, S, L)

    def forward(self, x):
        batch_size = x.size(0)
        V = x  # (B, G)

        V_tild   = V.unsqueeze(-1).unsqueeze(-1).expand(batch_size, V.size(1), self.S, self.L)
        I_i_tild = self.I_i.unsqueeze(0).expand(batch_size, -1, -1, -1)

        C = torch.sum(
            torch.prod(torch.gather(V_tild, 1, I_i_tild), dim=3),
            dim=2,
        )
        return C