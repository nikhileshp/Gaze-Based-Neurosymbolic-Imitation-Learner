import torch
import numpy as np
import matplotlib.pyplot as plt
import os

class GazeToMask():
    def __init__(self, N=84, sigmas=[10,10,10,10], coeficients = [1,1,1,1]):
        self.N = N
        assert len(sigmas) == len(coeficients)
        self.sigmas = sigmas
        self.coeficients = coeficients
        self.masks = self.initialize_mask()

    def generate_single_gaussian_tensor(self, map_size, mean_x, mean_y, sigma):
        x = torch.arange(map_size, dtype=torch.float32).unsqueeze(1).expand(map_size, map_size)
        y = torch.arange(map_size, dtype=torch.float32).unsqueeze(0).expand(map_size, map_size)
        # Calculate the Gaussian distribution for each element
        gaussian_tensor = (1 / (2 * torch.pi * sigma ** 2)) * torch.exp(
            -((x - mean_x) ** 2 + (y - mean_y) ** 2) / (2 * sigma ** 2))

        return gaussian_tensor

    def initialize_mask(self):
        temp_map = []
        N = self.N
        for i in range(len(self.sigmas)):
            temp = self.generate_single_gaussian_tensor(2 * N, N - 1, N - 1, self.sigmas[i])
            temp = temp/ temp.max()
            temp_map.append(self.coeficients[i]*temp)

        temp_map = torch.stack(temp_map, 0)

        return temp_map

    def find_suitable_map(self, Nx2=168, index=0, mean_x=0.5, mean_y=0.5):
        # returns a map such that the center of the gaussian is located at (mean_x, mean_y) of the map
        start_x, start_y = int((1 - mean_x) * Nx2 / 2), int((1 - mean_y) * Nx2 / 2)
        desired_map = self.masks[index][start_y:start_y + Nx2 // 2, start_x:start_x + Nx2 // 2]
        return desired_map

    def find_bunch_of_maps(self, means=[[0.5, 0.5]], offset_start=0):
        current_maps = torch.zeros([self.N, self.N])
        bunch_size = len(means)
        assert bunch_size + offset_start <= len(self.sigmas), f'The bunch is too long! It\'s length is {bunch_size}'
        Nx2 = self.N * 2
        for i in range(bunch_size):
            mean_x, mean_y = means[i][0], means[i][1]
            temp = self.find_suitable_map(Nx2, i+offset_start, mean_x, mean_y)
            current_maps = current_maps + temp

        return current_maps / torch.max(current_maps)

def get_gaze_mask(z_oreo, beta, target_size):
    z_oreo_flat = z_oreo.abs().sum(dim=1)
    # apply softmax on the last two dimensions of the tensor
    z_oreo_shape = z_oreo_flat.shape
    z_oreo_flat = z_oreo_flat.reshape(z_oreo.shape[0], -1)

    z_oreo_softmax = torch.nn.functional.softmax(z_oreo_flat / beta, dim=-1)
    z_oreo_softmax = z_oreo_softmax.reshape(z_oreo_shape)
    
    # scale up z_oreo to the size of xx
    z_oreo_mask = torch.nn.functional.interpolate(z_oreo_softmax.unsqueeze(1), size=target_size, mode='bicubic')

    # normalize z_oreo_mask
    flattened = z_oreo_mask.view(z_oreo_mask.shape[0], z_oreo_mask.shape[1], -1)
    gaze_mask = z_oreo_mask / flattened.max(dim=-1).values.unsqueeze(-1).unsqueeze(-1)

    return gaze_mask

def apply_gmd_dropout(z, g, test_mode = False):
    dropout_prob = 0.7
    B, C, H, W = z.shape
    A = torch.rand([B, 1, H, W], device=z.device)
    K = torch.nn.functional.interpolate(g, size=(H, W), mode='bicubic')
    K = (K - K.min()) / (K.max() - K.min())
    K = dropout_prob * K + (1 - dropout_prob)
    M = (A < K).float()
    if test_mode:
        z = z * K
    else:
        z = z * M
    return z

temp_flag = False
def plot_once(xx_clean, masked_xx, count, beta):
    global temp_flag
    os.makedirs("figs", exist_ok=True)
    if not temp_flag:
        idxs = np.random.choice(range(xx_clean.shape[0]), count)
        for idx in idxs:
            plt.imshow(masked_xx[idx, 0].cpu(), cmap='gray')
            plt.savefig("figs/obs_masked_{}_beta_{}.png".format(idx, beta), bbox_inches='tight')
            plt.clf()

            plt.imshow(xx_clean[idx, 0].cpu(), cmap='gray')
            plt.savefig("figs/obs_clean_{}_beta_{}.png".format(idx, beta), bbox_inches='tight')
            plt.clf()
        temp_flag = True
