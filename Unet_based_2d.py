from ntpath import join
import sys
import os 
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optimizor
from torch.autograd import Variable
from torchvision import datasets, transforms
from pydoc import locate
from tqdm import tqdm
import tensorboardX

from utils import * 
from models import * 
sys.path.append(r'../kitti_data')
from data import PointCloudFolder

## model config
from model_Unet import Unet, upsample
from SalsaNext_upsampling import SalsaNext


parser = argparse.ArgumentParser(description='VAE training of LiDAR')
parser.add_argument('--batch_size',         type=int,   default=64,             help='size of minibatch used during training')
parser.add_argument('--use_selu',           type=int,   default=0,              help='replaces batch_norm + act with SELU')
parser.add_argument('--base_dir',           type=str,   default='runs/test',    help='root of experiment directory')
parser.add_argument('--dataset_dir',        type=str,   default='../kitti_data/raw', help="path of dataset")
parser.add_argument('--no_polar',           type=int,   default=0,              help='if True, the representation used is (X,Y,Z), instead of (D, Z), where D=sqrt(X^2+Y^2)')
parser.add_argument('--lr',                 type=float, default=1e-3,           help='learning rate value')
parser.add_argument('--z_dim',              type=int,   default=128,            help='size of the bottleneck dimension in the VAE, or the latent noise size in GAN')
parser.add_argument('--autoencoder',        type=int,   default=1,              help='if True, we do not enforce the KL regularization cost in the VAE')
parser.add_argument('--atlas_baseline',     type=int,   default=0,              help='If true, Atlas model used. Also determines the number of primitives used in the model')
parser.add_argument('--panos_baseline',     type=int,   default=0,              help='If True, Model by Panos Achlioptas used')
parser.add_argument('--kl_warmup_epochs',   type=int,   default=150,            help='number of epochs before fully enforcing the KL loss')
parser.add_argument('--debug', action='store_true')

# ------------------------------------------------------------------------------
args = parser.parse_args()
maybe_create_dir(args.base_dir)
print_and_save_args(args, args.base_dir)

# the baselines are very memory heavy --> we split minibatches into mini-minibatches
if args.atlas_baseline or args.panos_baseline: 
    """ Tested on 12 Gb GPU for z_dim in [128, 256, 512] """ 
    bs = [4, 8 if args.atlas_baseline else 6][min(1, 511 // args.z_dim)]
    factor = args.batch_size // bs
    args.batch_size = bs
    is_baseline = True
    args.no_polar = 1
    print('using batch size of %d, ran %d times' % (bs, factor))
else:
    factor, is_baseline = 1, False

# reproducibility is good
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# model = NewCRF()
# construct model and ship to GPU

# 
name_struct = args.base_dir.split('/')[2]
assert len(name_struct.split('_')) == 5
model_type, repre, epoch, loss_type, _ = name_struct.split('_')

input_dim = input_dim_switch(repre)

############################# Select Network ###############################
if model_type == 'Salsa':
    model = SalsaNext(nclasses = input_dim)
elif model_type == 'Unet':
    model = Unet(imput_dim=input_dim,
                filters=64, 
                dropout_rate=0.25,
                upsampling_factor = 2)
else:
    raise ValueError
model.cuda()
#############################################################################
#############################################################################

# Logging
maybe_create_dir(os.path.join(args.base_dir, 'samples'))
maybe_create_dir(os.path.join(args.base_dir, 'models'))
writer = tensorboardX.SummaryWriter(log_dir=os.path.join(args.base_dir, 'TB'))
writes = 0
ns     = 80

# dataset preprocessing
print('loading data')
dataset_path = args.dataset_dir
# dataset_path = '../kitti_data/raw' 
print('path of dataset is {}'.format(os.path.join(os.getcwd(),dataset_path)))

dataset_train = PointCloudFolder(dataset_path, set='train', preprocess=preprocess)
dataset_val = PointCloudFolder(dataset_path, set='val', preprocess=preprocess)

# for item in train_loader:
#     print(item[0].shape)

if args.debug: 
    dataset_train, dataset_val = dataset_train[:128], dataset_val[:128]

# kitti_raw_path = os.path.join(os.getcwd(),data_path,'city','2011_09_26_drive_0005_sync')
# dataset_val_   = np.load(os.path.join(kitti_raw_path,"processed.npz") )
# dataset_train = preprocess(dataset_train).astype('float32') 
# dataset_val   = preprocess(dataset_val).astype('float32')

train_loader  = torch.utils.data.DataLoader(dataset_train, batch_size= args.batch_size,
                    shuffle=True, num_workers=0, drop_last=True)

val_loader    = torch.utils.data.DataLoader(dataset_val, batch_size= args.batch_size,
                    shuffle=False, num_workers=0, drop_last=False)

print(model)
optim = optimizor.Adam(model.parameters(), lr=args.lr) 

# build loss function
if args.atlas_baseline or args.panos_baseline:
    loss_fn = get_chamfer_dist()
else:
    # loss_fn = lambda a, b : (a - b).abs().sum(-1).sum(-1).sum(-1) 
    loss_fn = nn.L1Loss()


# VAE training
# ------------------------------------------------------------------------------

for epoch in range(100): # range(50 if args.autoencoder else 100):
    print('epoch %s' % epoch)
    model.train()
    loss_, elbo_, kl_cost_, kl_obj_, recon_ = [[] for _ in range(5)]

    # Output is always in polar, however input can be (X,Y,Z) or (D,Z)
    process_input = from_polar if args.no_polar else lambda x : x

    for i, img in enumerate(tqdm(train_loader)):
        img_inp = img[0].cuda()
        img_gt = img[1].cuda()

        if repre == 'xyz':
            img_inp, img_gt = from_polar(img_inp), from_polar(img_gt)
        elif repre == 'range':
            img_inp,_,_ = cart2sph_torch(from_polar(img_inp))
            img_gt, az_gt, elev_gt = cart2sph_torch(from_polar(img_gt))
        elif repre == 'polar':
            pass
        elif repre == 'rxyz':
            img_inp_r,_,_ = cart2sph_torch(from_polar(img_inp))
            img_gt_r,_,_ = cart2sph_torch(from_polar(img_gt))

            img_inp = torch.cat((img_inp_r,from_polar(img_inp)),dim=1)
            img_gt = torch.cat((img_gt_r,from_polar(img_gt)),dim=1)
        else:
            raise ValueError

        recon = model(img_inp)

        loss_recon = loss_fn(recon, img_gt)

        loss_    += [loss_recon.item()]

        # baseline loss is very memory heavy 
        # we accumulate gradient to simulate a bigger minibatch
        if (i+1) % factor == 0 or not is_baseline: 
            optim.zero_grad()

        loss_recon.backward()
        if (i+1) % factor == 0 or not is_baseline: 
            optim.step()

    writes += 1
    mn = lambda x : np.mean(x)
    print_and_log_scalar(writer, 'train/loss', mn(loss_), writes)
    loss_, elbo_, kl_cost_, kl_obj_, recon_ = [[] for _ in range(5)]
        
    # save some training reconstructions
    if epoch % 10 == 0:
         recon = recon[:ns].cpu().data.numpy()
         with open(os.path.join(args.base_dir, 'samples/train_{}.npz'.format(epoch)), 'wb') as f: 
             np.save(f, recon)

         print('saved training reconstructions')
         
    
    # Testing loop
    # --------------------------------------------------------------------------

    loss_, elbo_, kl_cost_, kl_obj_, recon_ = [[] for _ in range(5)]
    with torch.no_grad():
        model.eval()
        if epoch % 1 == 0:
            print('test set evaluation')
            for i, img in enumerate(val_loader):
                img_inp = img[0].cuda()
                img_gt = img[1].cuda()

                if repre == 'xyz':
                    img_inp, img_gt = from_polar(img_inp), from_polar(img_gt)
                elif repre == 'range':
                    img_inp,_,_ = cart2sph_torch(from_polar(img_inp))
                    img_gt, az_gt, elev_gt = cart2sph_torch(from_polar(img_gt))
                elif repre == 'polar':
                    pass
                elif repre == 'rxyz':
                    img_inp_r,_,_ = cart2sph_torch(from_polar(img_inp))
                    img_gt_r,_,_ = cart2sph_torch(from_polar(img_gt))

                    img_inp = torch.cat((img_inp_r,from_polar(img_inp)),dim=1)
                    img_gt = torch.cat((img_gt_r,from_polar(img_gt)),dim=1)
                else:
                    raise ValueError

                recon = model(img_inp)
           
                loss_recon = loss_fn(recon, img_gt)

                loss_    += [loss_recon.item()]

            print_and_log_scalar(writer, 'valid/loss', mn(loss_), writes)

            loss_, elbo_, kl_cost_, kl_obj_, recon_ = [[] for _ in range(5)]

            if epoch % 10 == 0:
                with open(os.path.join(args.base_dir, 'samples/test_{}.npz'.format(epoch)), 'wb') as f: 
                    recon = recon[:ns].cpu().data.numpy()
                    np.save(f, recon)
                    print('saved test recons')
               
                with open(os.path.join(args.base_dir, 'samples/sample_{}.npz'.format(epoch)), 'wb') as f: 
                    sample = process_input(img_inp).cpu().data.numpy()
                    np.save(f, sample)
                
                    print('saved model samples')

                with open(os.path.join(args.base_dir, 'samples/inputs_{}.npz'.format(epoch)), 'wb') as f: 
                    real = process_input(img_inp).cpu().data.numpy()
                    np.save(f, real)
                
                    print('saved real LiDAR')

    if (epoch + 1) % 10 == 0 :
        torch.save(model.state_dict(), os.path.join(args.base_dir, 'models/gen_{}.pth'.format(epoch)))


