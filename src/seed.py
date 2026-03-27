import torch
import numpy as np
from numpy import random
from src import config

def set_seed():
    
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)