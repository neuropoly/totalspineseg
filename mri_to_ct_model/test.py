import copy
from absl import app, flags
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import h5py
from diffusion import GaussianDiffusionSampler
from model import UNet
from Dataset.dataset import Valid_Data
from tqdm import tqdm


FLAGS = flags.FLAGS

# UNet
flags.DEFINE_integer('ch', 64, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 4, 4], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
flags.DEFINE_float('dropout', 0., help='dropout rate of resblock')

# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')

# Training
flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
flags.DEFINE_integer('img_size', 512, help='image size')
flags.DEFINE_integer('num_workers', 1, help='workers of Dataloader')

# Logging & Sampling
flags.DEFINE_string('DIREC', 'ddpm-unet', help='name of your project')
flags.DEFINE_integer('sample_size', 2, "sampling size of images")

device = torch.device('cuda:0')

def test():
    # dataset
    va_train = Valid_Data()
    validloader = DataLoader(va_train, batch_size=FLAGS.sample_size, num_workers=FLAGS.num_workers, 
                             pin_memory=True, shuffle=False)

    # model setup
    net_model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    ema_model = copy.deepcopy(net_model)

    
    ema_sampler = GaussianDiffusionSampler(
        ema_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type).to(device)
        
    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))
    
    checkpoint = torch.load('./Save/' + FLAGS.DIREC + '/model_latest.pkl')
    net_model.load_state_dict(checkpoint['net_model'])
    ema_model.load_state_dict(checkpoint['ema_model'])
    restore_epoch = checkpoint['epoch']
    print('Finish loading model')
                       
    output = np.zeros((27,5,512,512))  # example size, please change based on your data
    lr = np.zeros((27,512,512))
    hr = np.zeros((27,512,512))
    if not os.path.exists('Output/' + FLAGS.DIREC):
        os.makedirs('Output/' + FLAGS.DIREC)
    net_model.eval()
    count = 0
    with torch.no_grad():
        with tqdm(validloader, unit="batch") as tepoch:
            for data, target in tepoch:
                condition = data.to(device)                   
                length = data.shape[0]
                x_T = torch.randn(length, 1, FLAGS.img_size, FLAGS.img_size)
                x_T = x_T.to(device)
                x_0 = ema_sampler(x_T, condition)
                
                for kk in range(len(x_0)): # x_0 [10,b,1,h,w]
                    imgs = x_0[kk].data.squeeze().cpu() # imgs [b,h,w]
                    output[count:count+length,kk,:,:] = imgs              
                lr[count:count+length,:,:] = data.squeeze().cpu()
                hr[count:count+length,:,:] = target.squeeze().cpu()
                
                count += length
            
    path = 'Output/' + FLAGS.DIREC + '/result_epoch_' + str(restore_epoch) + '.hdf5'
    f = h5py.File(path, 'w')
    f.create_dataset('out', data=output)
    f.create_dataset('lr', data=lr)
    f.create_dataset('hr', data=hr)
    f.close()



def main(argv):
    test()


if __name__ == '__main__':
    app.run(main)
