import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import torch.nn.functional as F

from depth_anything_v2.dpt import DepthAnythingV2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--img-path', type=str, default="/root/catkin_ws/src/modules_vins/examples/DepthAnything/BotanicGarden-left.png")
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='/root/catkin_ws/src/modules_vins/examples/DepthAnything')
    
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
    

    with torch.no_grad():
        
        # depth = depth_anything.infer_image(raw_image, args.input_size)
        # image, (h, w) = depth_anything.image2tensor(raw_image, args.input_size)
        h = 518
        w = 924
        img = torch.rand(1, 3, h, w).to(DEVICE)
        depth = depth_anything(img)

        traced_script_module_for_zoeD_NK = torch.jit.trace(depth_anything, img)
        traced_script_module_for_zoeD_NK.save(f'checkpoints/depth_anything_v2_{args.encoder}_traced.pt')
