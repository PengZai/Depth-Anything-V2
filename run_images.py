import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from depth_anything_v2.dpt import DepthAnythingV2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--img-path', type=str, default="/media/spiderman/zhipeng_8t1/datasets/BotanicGarden/1018-00/1018_00_img10hz600p/left_rgb")
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='/media/spiderman/zhipeng_8t1/datasets/BotanicGarden/1018-00/1018_00_img10hz600p/tmp')
    
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
    
    os.makedirs(args.outdir, exist_ok=True)
    normalized_inv_depth_dir = os.path.join(args.outdir, 'normalized_inv_depth')
    os.makedirs(normalized_inv_depth_dir, exist_ok=True)

    inv_depth_for_vis_dir = os.path.join(args.outdir, 'inv_depth_for_vis')
    os.makedirs(inv_depth_for_vis_dir, exist_ok=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        timestamp = filename.split('.')[0].split('/')[-1]

        raw_image = cv2.imread(filename)
        
        inv_depth = depth_anything.infer_image(raw_image, args.input_size)
        detph = 1/(inv_depth+1e-9)

        inv_depth = (inv_depth - inv_depth.min()) / (inv_depth.max() - inv_depth.min())

        cv2.imwrite(os.path.join(normalized_inv_depth_dir, timestamp + '.tiff'), inv_depth)

        inv_depth_for_vis = inv_depth * 255.0
        inv_depth_for_vis = inv_depth_for_vis.astype(np.uint8)
        
        if args.grayscale:
            inv_depth_for_vis = np.repeat(inv_depth_for_vis[..., np.newaxis], 3, axis=-1)
        else:
            inv_depth_for_vis = (cmap(inv_depth_for_vis)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)


        cv2.imwrite(os.path.join(inv_depth_for_vis_dir, os.path.splitext(os.path.basename(filename))[0] + '_inv_depth_for_vis.png'), inv_depth_for_vis)
