
import os
import random

import numpy as np
import torch

def set_seeds(seed: int | None = None, force: bool = False, warn_only: bool = True ) -> int:
    """
    Set the random seed using the required value (`seed`). 
    The seed affects the following libraries: random, numpy, torch.

    Args:
        seed (int, optional): random seed value to set; if None, a random seed will be generated.
            
        force (bool, optional, default = False):  if True cuDNN will be set to benchmark mode, and Torch will use deterministic algorithms. cuDNN benchmarks multiple convolution algorithms and select the fastest. This mode is good whenever the input sizes for your network do not vary. In case of changing input size, cuDNN will benchmark every time a new input size appears, which will probably lead to worse performance. If you want perfect reproducibility, you should set this to False. 

        warn_only (bool, optional, default = True): If True, operations that do not have a deterministic implementation will throw a warning instead of an error.
    Returns:
        the newly set seed
    """

    if force:
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        torch.backends.cudnn.deterministic = True
        
    if seed is None:
        seed = random.randint(1, 10000)
        
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    return seed