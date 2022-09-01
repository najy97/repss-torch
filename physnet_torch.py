# # import package
#
# # model
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchsummary import summary
# from torch import optim
#
# # dataset and transformation
# from torchvision import datasets
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from torchvision import models
# import os
#
# # display images
# from torchvision import utils
# import matplotlib.pyplot as plt
#
# # utils
# import numpy as np
# from torchsummary import summary
# import time
# import copy
import math


def neg_pear_loss(t, x, y):
    sum_xy = sum_x = sum_y = sum_x_sq = sum_y_sq = 0

    for i in range(t):
        sum_xy += x[i] * y[i]
        sum_x += x[i]
        sum_y += y[i]
        sum_x_sq += math.pow(x[i], 2)
        sum_y_sq += math.pow(y[i], 2)

    num = t * sum_xy - sum_x * sum_y
    den = math.sqrt((t * sum_x_sq - math.pow(sum_x, 2)) * ((t * sum_y_sq) - math.pow(sum_y, 2)))
    loss = 1 - num / den
    return loss


