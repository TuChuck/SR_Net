from contextlib import AbstractAsyncContextManager
from torchvision import datasets, transforms
import torch.utils.data
import torch
import sys

import __init__

# metric import
from EMD.emd_module import emdModule as emd
from Chamfer_distance.chamfer_distance import ChamferDistance

from utils import * 
from models import * 
from tqdm import tqdm

# dataset loader
sys.path.append(r'../kitti_data')
from data import PointCloudFolder

## model load
model_path = "../" + sys.argv[1]
# sys.path.append("../" + model_config)
from model_Unet import Unet, upsample
from SalsaNext_upsampling import SalsaNext

'''
Expect two arguments: 
    1) path_to_model_folder
    2) epoch of model you wish to load
    3) metric to evaluate on 
e.g. python eval.py runs/test_baseline 149 emd
'''

# reproducibility is good
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

nb_samples = 200
out_dir = os.path.join(model_path, 'final_samples')
maybe_create_dir(out_dir)
save_test_dataset = False
size = 5
fast = True

#
name_struct = sys.argv[1].split('/')[1]
assert len(name_struct.split('_')) == 5
model_type, repre, epoch, loss_type, _ = name_struct.split('_')

input_dim = input_dim_switch(repre)

# fetch metrics
if 'emd' in sys.argv[3]:
    # https://github.com/Colin97/MSN-Point-Cloud-Completion
    loss = emd
elif 'chamfer' in sys.argv[3]:
    # https://github.com/chrdiller/pyTorchChamferDistance
    loss = ChamferDistance
else:
    raise ValueError("{} is not a valid metric for point cloud eval. Either \'emd\' or \'chamfer\'"\
            .format(sys.argv[2]))

with torch.no_grad():

    # 1) load trained model
    if model_type == 'Salsa':
        model = SalsaNext(nclasses = input_dim)
    elif model_type == 'Unet':
        model = Unet(imput_dim=input_dim,
                 filters=64, 
                 dropout_rate=0.25,
                 upsampling_factor = 2)
    else:
        raise ValueError

    model.load_state_dict(torch.load(os.path.join(model_path,"models","gen_"+sys.argv[2]+".pth")))
    model.cuda()
    
    # 2) load data
    print('test set reconstruction')
    dataset_train = PointCloudFolder('../kitti_data/raw', set='train', preprocess=True)
    dataset_test = PointCloudFolder('../kitti_data/raw', set='test', preprocess=True)
    # dataset_test = preprocess(dataset_test).astype('float32')

    loader = (torch.utils.data.DataLoader(dataset_test, batch_size=size,
                        shuffle=False, num_workers=0, drop_last=True))

    loss_fn = loss()

    # missing reconstruction
    # for missing in [.97, .98, .99, .999]:#[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45][::(2 if fast else 1)]:
    for missing in [.0]:
        losses = []
        si = 0
        for batch_ in tqdm(loader):
            inp_tensor = batch_[0].cuda()
            gt_tensor = batch_[1].cuda()

            if repre == 'xyz':
                inp_tensor, gt_tensor = from_polar(inp_tensor), from_polar(gt_tensor)
            elif repre == 'range':
                inp_tensor,_,_ = cart2sph_torch(from_polar(inp_tensor))
                gt_tensor, az_gt, elev_gt = cart2sph_torch(from_polar(gt_tensor))
            elif repre == 'polar':
                pass
            elif repre == 'rxyz':
                inp_tensor_r,_,_ = cart2sph_torch(from_polar(inp_tensor))
                gt_tensor_r,_,_ = cart2sph_torch(from_polar(gt_tensor))

                inp_tensor = torch.cat((inp_tensor_r,from_polar(inp_tensor)),dim=1)
                gt_tensor = torch.cat((gt_tensor_r,from_polar(gt_tensor)),dim=1)
            else:
                raise ValueError

            recon = model(inp_tensor)

            if 'emd' in sys.argv[3]:
                if repre == 'xyz':
                    recon_3d = recon
                    gt_3d = gt_tensor
                elif repre == 'range':
                    recon_3d = sph2cart_torch(recon, az_gt, elev_gt)

                    gt_3d = sph2cart_torch(gt_tensor, az_gt, elev_gt)

                elif repre == 'polar':
                    recon_3d = from_polar(recon)
                    gt_3d = from_polar(gt_tensor)
                elif repre == 'rxyz':
                    recon_3d = recon[:,1:]
                    gt_3d = gt_tensor[:,1:]
                else:
                    raise ValueError

                recon_3d = recon_3d.permute(0,2,3,1).view(size, -1, 3)
                gt_3d = gt_3d.permute(0,2,3,1).view(size, -1, 3)
                
                norm_recon_3d = (recon_3d + torch.abs(recon_3d.min()))
                norm_recon_3d /= norm_recon_3d.max()

                norm_gt_3d = (gt_3d + torch.abs(gt_3d.min()))
                norm_gt_3d /= norm_gt_3d.max()

                dist, _ = loss_fn(recon_3d, gt_3d,eps=0.005, iters = 100)   ## (pred, gt)
                emd = torch.sqrt(dist).mean()
                losses += [emd]
            elif 'chamfer' in sys.argv[3]:
                if repre == 'xyz':
                    recon_3d = recon
                    gt_3d = gt_tensor
                elif repre == 'range':
                    recon_3d = sph2cart_torch(recon, az_gt, elev_gt)

                    gt_3d = sph2cart_torch(gt_tensor, az_gt, elev_gt)

                elif repre == 'polar':
                    recon_3d = from_polar(recon)
                    gt_3d = from_polar(gt_tensor)

                elif repre == 'rxyz':
                    recon_3d = recon[:,:3]
                    gt_3d = gt_tensor[:,:3]
                else:
                    raise ValueError

                recon_3d = recon_3d.permute(0,2,3,1).view(size, -1, 3)
                gt_3d = gt_3d.permute(0,2,3,1).view(size, -1, 3)

                dist1, dist2 = loss_fn(gt_3d, recon_3d)                       ## (gt, pred)
                losses += [torch.mean(dist1) + torch.mean(dist2)]

            elif 'RMSE' in sys.argv[3] :
                dist = torch.mean(torch.sqrt(recon_xyz * recon_xyz + from_polar(gt_tensor) * from_polar(gt_tensor)))
                losses += [dist]
        
        losses = torch.stack(losses).mean().item()
        print('{} with missing p {} : {:.4f}'.format(sys.argv[3], missing, losses))
