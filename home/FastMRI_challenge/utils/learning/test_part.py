import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
from utils.model.varnet import VarNet
from utils.model.varnet import AccConvertModule

def test(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    
    with torch.no_grad():
        for (mask, kspace, _, _, fnames, slices, _ ) in data_loader:            
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)            
            output = model(kspace, mask)

            # for batch
            for i in range(output.shape[0]):                
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
    
    for fname in reconstructions:                
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )        

        
    return reconstructions, None


def forward(args):

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print ('Current cuda device ', torch.cuda.current_device())

    if(args.module == 0):
        model = VarNet(
                    module_path=None,
                    num_cascades=args.cascade, 
                    chans=args.chans, 
                    sens_chans=args.sens_chans)
    else:
        model = AccConvertModule(
                    num_cascades=args.cascade, 
                    chans=args.chans, 
                    sens_chans=args.sens_chans)
    model.to(device=device)
    
    checkpoint = torch.load(args.exp_dir / 'model_7_8_27.pt', map_location='cpu')
    print(checkpoint['epoch'], checkpoint['best_val_loss'].item())
    model.load_state_dict(checkpoint['model'])
    
    
    forward_loader = create_data_loaders(data_path = args.data_path, args = args, isforward = True)
    print("test start !")
    reconstructions, _ = test(args, model, forward_loader)
    print("saving start !")
    save_reconstructions(reconstructions, args.forward_dir)