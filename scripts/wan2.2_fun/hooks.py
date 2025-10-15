import torch
from torch.nn import functional as F

interpolate = F.interpolate

def interpolate_hook(*args, **kwargs):
    if kwargs['mode'] == 'trilinear':
        if args[0].dtype == torch.bfloat16:
            args = list(args)
            args[0] = args[0].to(torch.float16)
            args = tuple(args)
            out = interpolate(*args, **kwargs)
            return out.to(torch.bfloat16)
        else:
            return interpolate(*args, **kwargs)
    else:
        return interpolate(*args, **kwargs)

F.interpolate = interpolate_hook