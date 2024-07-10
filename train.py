import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from PIL import Image
from PIL import ImageFile
from torchvision import transforms
from torchvision.utils import save_image
import time
import cv2 
from tqdm.auto import tqdm
from sampler import InfiniteSamplerWrapper

from math import log, sqrt, pi

from styleaug import distill
from styleaug.distill import Distiller

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated


def train_transform():
    transform_list = [
        transforms.ToTensor(),
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = os.listdir(self.root)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'

IMAGE_EXTENSIONS = ('.ras', '.xwd', '.bmp', '.dib', '.jpe', '.jpg', '.jpeg', '.xpm', '.ief', '.pbm', 
                       '.tif', '.gif', '.ppm', '.xbm', '.tiff', '.rgb', '.pgm', '.png', '.pnm', '.webp', 
                       '.ico', '.exr', '.avif', '.hdr', '.pic')

class FolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FolderDataset, self).__init__()
        self.root = root
        self.paths = list(self.get_paths(self.root))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'RecursiveFolderDataset'
    
    def get_paths(self, root: str, selected_extensions = IMAGE_EXTENSIONS):
        # legits_path = []
        for path in os.listdir(root):
            if os.path.isfile(os.path.join(root, path)) and path.lower().endswith(selected_extensions):
                yield os.path.join(root, path)
            elif os.path.isdir(os.path.join(root, path)):
                yield from self.get_paths(os.path.join(root, path), selected_extensions)
            else:
                continue


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, required=True,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, required=True,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')

# training options
parser.add_argument('--save_dir', default='styleaug/checkpoints',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=4)

parser.add_argument('--mse_weight', type=float, default=0)
parser.add_argument('--style_weight', type=float, default=1)
parser.add_argument('--content_weight', type=float, default=0.1)

# save options
parser.add_argument('--n_threads', type=int, default=8)
parser.add_argument('--print_interval', type=int, default=100)
parser.add_argument('--save_model_interval', type=int, default=5000)
parser.add_argument('--start_iter', type=int, default=0, help='starting iteration')
parser.add_argument('--resume', default="distilled.pth", type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

if __name__ == '__main__':
    args = parser.parse_args()

        
    device = torch.device('cuda')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok= True)
    args.resume = os.path.join(args.save_dir, args.resume)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok= True)


    # glow
    distiller = Distiller()

    # -----------------------resume training------------------------
    if args.resume:
        if os.path.isfile(args.resume):
            print("--------loading checkpoint----------")
            print("=> loading checkpoint '{}'".format(args.resume))
            distiller.load_ckpt(args.resume)
            #optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("--------no checkpoint found---------")
    distiller = distiller.to(device)
    distiller = nn.DataParallel(distiller)
    distiller.train()



    # -------------------------------------------------------------
    content_tf = train_transform()
    style_tf = train_transform()

    content_dataset = FolderDataset(args.content_dir, content_tf)
    style_dataset = FolderDataset(args.style_dir, style_tf)

    content_iter = iter(data.DataLoader(
        content_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=args.n_threads))
    style_iter = iter(data.DataLoader(
        style_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=args.n_threads))

    optimizer = torch.optim.Adam(distiller.module.parameters(), lr=args.lr)

    log_c = []
    log_s = []
    log_mse = []
    Time = time.time()
    # -----------------------training------------------------
    for i in tqdm(range(args.start_iter, args.max_iter)):
        adjust_learning_rate(optimizer, iteration_count=i)
        content_images = next(content_iter).to(device)
        style_images = next(style_iter).to(device)

        # glow forward: real -> z_real, style -> z_style
        if i == args.start_iter:
            with torch.no_grad():
                _ = distiller.module(content_images, forward=True)
                continue

        # (log_p, logdet, z_outs) = glow()
        t_content, s_content, t_style, s_style = distiller(content_images, style_images)
        loss = distiller.module.compute_loss(content_images, style_images, t_content, s_content, t_style, s_style)


        # optimizer update
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(distiller.module.parameters(), 5)
        optimizer.step()
        
        # update loss log
        log_c.append(loss.item())

        # save image
        if i % args.print_interval == 0:

            output_name = os.path.join(args.save_dir, "%06d.jpg" % i)
            output_images = torch.cat((content_images.cpu(), style_images.cpu(), t_content.cpu(), 
                                        s_content.cpu()), 
                                    0)
            save_image(output_images, output_name, nrow=args.batch_size)
            
            print("iter %d   time/iter: %.2f   loss: %.3f  " % (i, 
                                                                        (time.time()-Time)/args.print_interval, 
                                                                        np.mean(np.array(log_c))
                                                                        ))
            log_c = []
            Time = time.time()

            
        if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
            distiller.module.save_ckpt(args.resume)


