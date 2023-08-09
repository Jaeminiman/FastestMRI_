import argparse
from pathlib import Path
import os, sys
if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')

from utils.learning.test_part import forward

    
def parse():
    parser = argparse.ArgumentParser(description='Test Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU_NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-n', '--net_name', type=Path, default='test_varnet', help='Name of network')
    #parser.add_argument('-p', '--path_data', type=Path, default='/Data/leaderboard/', help='Directory of test data')
    parser.add_argument('-t', '--path_data', type=Path, default='/workspace/Fastest/Data/leaderboard', help='Directory of train data')
    
    parser.add_argument('--m', type=int, default=0, help='Is this for module? | 0 : total model, 1 : module')
    parser.add_argument('--cascade', type=int, default=2, help='Number of cascades | Should be less than 12')
    parser.add_argument('--chans', type=int, default=9, help='Number of channels for cascade U-Net')
    parser.add_argument('--sens_chans', type=int, default=4, help='Number of channels for sensitivity map U-Net')
    parser.add_argument("--input-key", type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    
    args.exp_dir = '../result' / args.net_name / 'checkpoints_module_V2'
    # args.exp_dir = '../result' / args.net_name / 'checkpoints'
    
    # # acc4
    # args.data_path = args.path_data / "acc4"
    # args.forward_dir = '../result' / args.net_name / 'reconstructions_leaderboard' / "acc4"
    # print(args.forward_dir)
    # forward(args)
    
    # acc8
    args.data_path = args.path_data / "acc8"
    args.forward_dir = '../result' / args.net_name / 'reconstructions_leaderboard_module_V2' / "acc8"
    print(args.forward_dir)
    forward(args)
    
