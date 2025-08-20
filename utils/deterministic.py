
import os
import random

import numpy as np
import torch

def set_seed(seed: int | None = None, force: bool = False ) -> int:
    """
    Set the random seed using the required value (`seed`). 

    The seed affects the following libraries:
    - random
    - numpy
    - torch

    Args:
        seed: random seed value to set;
            if `None`, a random seed will be generated
            
        force: if 'true' cuDNN will be set to benchmark mode, and torch will use deterministic algorithms.

    Returns:
        the newly set seed
    """
        # --- enable cuDNN benchmark:
    # cuDNN benchmarks multiple convolution algorithms and select the fastest.
    # This mode is good whenever the input sizes for your network do not vary.
    # In case of changing input size, cuDNN will benchmark every time a new
    # input size appears, which will probably lead to worse performance. If you
    # want perfect reproducibility, you should set this to `False`
    #If you want to have reproducible results, you should also set the
    # following flags to `True`. This will make sure that the algorithms used
    # by PyTorch and cuDNN are deterministic.
    
    if force:
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        
        
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    return seed