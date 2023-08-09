import numpy as np
import torch

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)

class DataTransform:
    def __init__(self, isforward, max_key):
        self.isforward = isforward
        self.max_key = max_key
    def __call__(self, mask, input, target, attrs, fname, slice, mask_len = 0):                
        if not self.isforward:           
            target = to_tensor(target)     
            maximum = attrs[self.max_key]
        else:            
            target = -1
            maximum = -1

        if(mask_len == 392):            
            pad_width = 2
            input = np.pad(input, ((0,0), (0, 0), (pad_width, pad_width)), mode='edge') 
            assert(input.shape[-1] == 396)
            
        kspace = to_tensor(input * mask)
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)        
        mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
        return mask, kspace, target, maximum, fname, slice, mask_len

class DataTransform_Module:
    def __init__(self, isforward, max_key):
        self.isforward = isforward        
        self.max_key = max_key
    def __call__(self, mask_4x, mask_8x, target, kspace, attrs, fname, slice, mask_len = 0):                        
        if not self.isforward:         
            target = to_tensor(target)               
            maximum = attrs[self.max_key]
        else:                        
            target = -1
            maximum = -1

        if(mask_len == 392):            
            pad_width = 2
            kspace = np.pad(kspace, ((0,0), (0, 0), (pad_width, pad_width)), mode='edge') 
            assert(kspace.shape[-1] == 396)

        kspace_input = to_tensor(kspace * mask_8x)    
        kspace_target = to_tensor(kspace * mask_4x)        
        kspace_input = torch.stack((kspace_input.real, kspace_input.imag), dim=-1)
        kspace_target = torch.stack((kspace_target.real, kspace_target.imag), dim=-1)
        mask_8x = torch.from_numpy(mask_8x.reshape(1, 1, kspace_input.shape[-2], 1).astype(np.float32)).byte()
        return mask_8x, target, kspace_input, kspace_target, maximum, mask_len
