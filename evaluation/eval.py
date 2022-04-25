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
import tqdm

# dataset loader
sys.path.append(r'../kitti_data')
from data import PointCloudFolder


## model load


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
out_dir = os.path.join(sys.argv[1], 'final_samples')
maybe_create_dir(out_dir)
save_test_dataset = False
size = 10 if 'emd' in sys.argv[3] else 5
fast = True

# fetch metrics
# if 'emd' in sys.argv[3]: 
#     loss = emd.emdModule
# elif 'chamfer' in sys.argv[3]:
#     loss = get_chamfer_dist:
# else:
#     raise ValueError("{} is not a valid metric for point cloud eval. Either \'emd\' or \'chamfer\'"\
#             .format(sys.argv[2]))

with torch.no_grad():

    # 1) load trained model
    # model = load_model_from_file(sys.argv[1], epoch=int(sys.argv[2]), model='gen')[0]
    # model = model.cuda()
    # model.eval()

    model = CRF_based_LSRNet(input_dim=3)
    model.load_state_dict(torch.load(os.path.join(sys.argv[1],"models","gen_"+sys.argv[2]+".pth")))
    model.cuda()
    if 'panos' in sys.argv[1] or 'atlas' in sys.argv[1] : model.args.no_polar = 1 
    
    # 2) load data
    print('test set reconstruction')
    # dataset = np.load('./kitti_data/lidar_val.npz')
    dataset_train = PointCloudFolder('./kitti_data/raw', set='train', preprocess=True)
    dataset_test = PointCloudFolder('./kitti_data/raw', set='test', preprocess=True)#preprocess=preprocess)
    # dataset_test = preprocess(dataset_test).astype('float32')

    # if fast: dataset_test = dataset_test[:100]
    
    # if save_test_dataset: 
    #     np.save(os.path.join(out_dir, 'test_set'), dataset)

    loader = (torch.utils.data.DataLoader(dataset_test, batch_size=size,
                        shuffle=True, num_workers=4, drop_last=True)) #False))

    # loss_fn = loss()
    #######  H9  #######
    # process_input = (lambda x : x) if model.args.no_polar else to_polar
    process_input = to_polar
    ####################

    # noisy reconstruction
    for noise in []:#0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.][::(2 if fast else 1)]: 
        losses = []
        for batch in loader:
            batch = batch.cuda() 
            batch_xyz = from_polar(batch)
            noise_tensor = torch.zeros_like(batch_xyz).normal_(0, noise)

            means = batch_xyz.transpose(1,0).reshape((3, -1)).mean(dim=-1)
            stds  = batch_xyz.transpose(1,0).reshape((3, -1)).std(dim=-1)
            means, stds = [x.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) for x in [means, stds]]

            # normalize data
            norm_batch_xyz = (batch_xyz - means) / (stds + 1e-9)
            # add the noise
            input = norm_batch_xyz + noise_tensor

            # unnormalize
            input = input * (stds + 1e-9) + means

            recon = model(process_input(input))[0]
            recon_xyz = from_polar(recon)

            if 'emd' in sys.argv[3]:
                losses += [loss_fn(recon_xyz, batch_xyz,ep=0.0002, k = 10000)]
            elif 'chamfer' in sys.argv[3]:
                losses += [loss_fn(recon_xyz, batch_xyz)]

        losses = torch.stack(losses).mean().item()
        print('{} with noise {} : {:.4f}'.format(sys.argv[3], noise, losses))

        del input, recon, recon_xyz, losses

    ####### H9 ########
    # process_input = from_polar if model.args.no_polar else (lambda x : x)
    process_input = from_polar
    ##################
    # missing reconstruction
    # for missing in [.97, .98, .99, .999]:#[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45][::(2 if fast else 1)]:
    for missing in [.0]:
        losses = []
        si = 0
        for batch_ in loader:
            si = si+1
            ptg = si / len(loader) * 100
            print(">>>>> {} % >>>>>>".format(ptg))
            inp_tensor = batch_[0].cuda()
            gt_tensor = batch_[1].cuda()
            inp_tensor_xyz = from_polar(inp_tensor)

            # is_present = (torch.zeros_like(inp_tensor[:, [0]]).uniform_(0,1) + (1 - missing)).floor()
            # is_present = (torch.zeros_like(inp_tensor[:, [0]]))

            input = inp_tensor_xyz #* is_present
            
            # SMOOTH OUT ZEROS
            if missing > 0: input = torch.Tensor(remove_zeros(input)).float().cuda()

            recon = model(process_input(input).float().cuda())
            recon_xyz = from_polar(recon)

            # TODO: remove this
            #recon_xyz[:, 0].uniform_(batch_xyz[:, 0].min(), batch_xyz[:, 0].max())
            #recon_xyz[:, 1].uniform_(batch_xyz[:, 1].min(), batch_xyz[:, 1].max())
            #recon_xyz[:, 2].uniform_(batch_xyz[:, 2].min(), batch_xyz[:, 2].max())

            if 'emd' in sys.argv[3]:
                recon_3D = recon_xyz.permute(0,3,2,1).reshape(size,-1,3)
                batch_3D = batch_xyz.permute(0,3,2,1).reshape(size,-1,3)
                
                dist, _ = loss_fn(recon_3D, batch_3D,eps=0.0002, iters = 10000)
                emd = torch.sqrt(dist).mean()
                losses += [emd]
            elif 'chamfer' in sys.argv[3]:
                losses += [loss_fn(recon_xyz, from_polar(gt_tensor))]

            elif 'RMSE' in sys.argv[3] :
                dist = torch.mean(torch.sqrt(recon_xyz * recon_xyz + from_polar(gt_tensor) * from_polar(gt_tensor)))
                losses += [dist]
        
        losses = torch.stack(losses).mean().item()
        print('{} with missing p {} : {:.4f}'.format(sys.argv[3], missing, losses))
