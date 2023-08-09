import h5py
import random
from utils.data.transforms import DataTransform
from utils.data.transforms import DataTransform_Module
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np

class Mask:
    def __init__(self):
        self.acc4 = self.make_acc4()
        self.acc6 = self.make_acc6()
        self.acc8 = self.make_acc8()
    def getMask(self, maskIdx):
        if(maskIdx == "4x"):
            return self.acc4
        elif(maskIdx == "8x"):
            return self.acc8
        elif(maskIdx == "6x"):
            return self.acc6
        else:    
            raise NotImplementedError("maskIdx가 잘못됨")
    def make_acc4(self):
        n = 396
        arr = np.zeros((n,), dtype = np.float32)
        arr[2::4] = 1
        arr[n//2 - 33//2: 33 + n//2 - 33//2] = 1        
        return arr
    
    def make_acc6(self):
        n = 396
        arr = np.zeros((n,), dtype = np.float32)
        arr[4::6] = 1
        arr[n//2 - 33//2: 33 + n//2 - 33//2] = 1        
        return arr
    
    def make_acc8(self):
        n = 396
        arr = np.zeros((n,), dtype = np.float32)        
        arr[6::8] = 1                
        arr[n//2 - 33//2: 33 + n//2 - 33//2] = 1
        return arr


# forward : True이면 -> image 데이터(Ground truth) 이용 x -> 즉, inference 용(test)으로 쓰려면 forwad == true 인 듯
class SliceData(Dataset):
    def __init__(self, root, transform, input_key="kspace", target_key="image_label", forward=False):
        self.transform = transform
        self.input_key = input_key # kspace
        self.target_key = target_key # image_label
        self.forward = forward
        self.mask = Mask()
        self.image_examples = []
        self.kspace_examples = []

        # maskIdxs = ["4x", "6x", "8x"]
        maskIdxs = ["4x", "8x"]
        # for training -> mask 여러개 data augmentation
        if not forward:
            image_files = list(Path(root / "image").iterdir())
            for fname in sorted(image_files):
                for maskIdx in maskIdxs:
                    num_slices, _ = self._get_metadata(fname)

                    self.image_examples += [
                        (fname, slice_ind) for slice_ind in range(num_slices)
                    ]

            kspace_files = list(Path(root / "kspace").iterdir())
            for fname in sorted(kspace_files):
                for maskIdx in maskIdxs:

                    num_slices, mask_len = self._get_metadata(fname)
                    

                    self.kspace_examples += [
                        (fname, slice_ind, maskIdx, mask_len) for slice_ind in range(num_slices)
                    ]
        else :
            kspace_files = list(Path(root / "kspace").iterdir())
            for fname in sorted(kspace_files):                

                num_slices, _ = self._get_metadata(fname)

                self.kspace_examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]

    # num_slices : slice 개수
    # length : mask 길이가 396인지 392인지 결정
    def _get_metadata(self, fname):
        length = None
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
            if("mask" in hf.keys()):
                length = len(hf["mask"])
        return num_slices, length
     
    def __len__(self):
        return len(self.kspace_examples)


    def __getitem__(self, i):    
        mask_len = 0    
        if not self.forward:
            image_fname, dataslice_i = self.image_examples[i]
            kspace_fname, dataslice, maskIdx, mask_len = self.kspace_examples[i]            

            # consistency test
            assert(dataslice_i == dataslice)
            assert(str(image_fname).split("/")[-1] == str(kspace_fname).split("/")[-1])
        else:            
            kspace_fname, dataslice = self.kspace_examples[i]
        
        
            
        
        with h5py.File(kspace_fname, "r") as hf:
            input = hf[self.input_key][dataslice]
            if not self.forward:
                mask = self.mask.getMask(maskIdx)
            else:                
                mask =  np.array(hf["mask"])
            
                      
        if self.forward:
            target = -1
            attrs = -1
        else:            
            with h5py.File(image_fname, "r") as hf:                
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)
            
        return self.transform(mask, input, target, attrs, kspace_fname.name, dataslice, mask_len)


def create_data_loaders(data_path, args, shuffle=False, isforward=False):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1

    data_storage = SliceData(
        root=data_path,
        transform=DataTransform(isforward, max_key_),
        input_key=args.input_key,
        target_key=target_key_,
        forward = isforward
    )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
    )
    return data_loader


############################################################################################################
# Dataset for acc 8 -> acc 4 module training
############################################################################################################


class SliceData_Module(Dataset):
    def __init__(self, root, transform, input_key="kspace", target_key = "image_label", forward=False):
        self.transform = transform
        self.input_key = input_key # kspace
        self.target_key = target_key # image_label
        self.forward = forward
        self.mask = Mask()        
        self.kspace_examples = []
        self.image_examples = []
        
               
        if not forward:
            image_files = list(Path(root / "image").iterdir())
            for fname in sorted(image_files):                
                num_slices, _ = self._get_metadata(fname)

                self.image_examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]

            kspace_files = list(Path(root / "kspace").iterdir())
            for fname in sorted(kspace_files):                
                num_slices, mask_len = self._get_metadata(fname)                    

                self.kspace_examples += [
                    (fname, slice_ind, mask_len) for slice_ind in range(num_slices)
                ]
        else :                        
            kspace_files = list(Path(root / "kspace").iterdir())
            for fname in sorted(kspace_files):                

                num_slices, _ = self._get_metadata(fname)

                self.kspace_examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]

    # num_slices : slice 개수
    # length : mask 길이가 396인지 392인지 결정
    def _get_metadata(self, fname):
        length = None
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
            if("mask" in hf.keys()):
                length = len(hf["mask"])
        return num_slices, length
     
    def __len__(self):
        return len(self.kspace_examples)


    def __getitem__(self, i):    
        mask_len = 0    
        if not self.forward:            
            image_fname, dataslice_i = self.image_examples[i]
            kspace_fname, dataslice, mask_len = self.kspace_examples[i]            

            # consistency test
            assert(dataslice_i == dataslice)
            assert(str(image_fname).split("/")[-1] == str(kspace_fname).split("/")[-1])            
        else:            
            kspace_fname, dataslice = self.kspace_examples[i]
        
                
        with h5py.File(kspace_fname, "r") as hf:
            kspace = hf[self.input_key][dataslice]
            
        mask_4x = self.mask.getMask("4x")
        mask_8x = self.mask.getMask("8x")            


        if self.forward:            
            target = -1
            attrs = -1
        else:            
            with h5py.File(image_fname, "r") as hf:                
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)
        
        return self.transform(mask_4x, mask_8x, target, kspace, attrs, kspace_fname.name, dataslice, mask_len)

def create_data_loaders_Module(data_path, args, shuffle=False, isforward=False):    
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1

    data_storage = SliceData_Module(
        root=data_path,
        transform=DataTransform_Module(isforward, max_key_),
        input_key=args.input_key,
        target_key=target_key_,
        forward = isforward
    )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
    )
    return data_loader
