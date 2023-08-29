import torch
import torch.utils.data
import numpy as np
import OpenEXR, Imath, array
from torchvision import transforms
import scipy.io
import math
import os.path as osp
import cv2
import glob


def random_uniform(low, high, size):
    n = (high - low) * torch.rand(size) + low
    return n.numpy()


class M3D_Dataset(torch.utils.data.Dataset):
    '''PyTorch dataset module for effiicient loading'''

    def __init__(self,
                 root_path,
                 path_to_img_list,
                 rotate=False,
                 flip=False,
                 permute_color=False):

        # Set up a reader to load the panos
        self.root_path = root_path
        self.image_list = self.build_list()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


        # Max depth for GT
        self.max_depth = 8.0
        self.min_depth = 0.1
        self.rotate = rotate
        self.flip = flip
        self.permute_color = permute_color
        self.pano_w = 1024
        self.pano_h = 512

    def build_list(self):
        path_parent = glob.glob("/home/lgj/360/dataset360/pano3d/mnt/cache/chuguanyi1/fisheye/full/*")
        path_img_all = []
        # import ipdb
        # ipdb.set_trace()
        for i in path_parent:
            path_img = glob.glob(i + "/*.png")
            path_img_all += path_img
        return path_img_all

    def __getitem__(self, idx):
        '''Load the data'''

        # Select the panos to load
        img_path = self.image_list[idx]
        # import ipdb
        # ipdb.set_trace()
        # Load the panos
        # relative_basename = osp.splitext((relative_paths[0]))[0]
        # basename = osp.splitext(osp.basename(relative_paths[0]))[0]
        rgb = self.readRGBPano(img_path)
        # depth = self.readDepthPano(self.root_path + relative_paths[3])
        depth_path = img_path.split("emission_center_0")[0] + 'depth_center_0.exr'
        try:
            # import ipdb
            # ipdb.set_trace()
            depth = self.readDepthPano(depth_path)
            print(depth.shape)

        except:
            depth = np.random.rand(512,1024)
        rgb = rgb.astype(np.float32) / 255

        # Random flip
        if self.flip:
            if torch.randint(2, size=(1,))[0].item() == 0:
                rgb = np.flip(rgb, axis=1)
                depth = np.flip(depth, axis=1)

        # Random horizontal rotate
        if self.rotate:
            dx = torch.randint(rgb.shape[1], size=(1,))[0].item()
            dx = dx // (rgb.shape[1] // 4) * (rgb.shape[1] // 4)
            rgb = np.roll(rgb, dx, axis=1)
            depth = np.roll(depth, dx, axis=1)

        # Random gamma augmentation
        if self.permute_color:
            if torch.randint(4, size=(1,))[0].item() == 0:
                idx = np.random.permutation(3)
                rgb = rgb[:, :, idx]

        depth = np.expand_dims(depth, 0)
        depth_mask = ((depth <= self.max_depth) & (depth > self.min_depth)).astype(np.uint8)

        # Threshold depths
        depth *= depth_mask
        # Convert to torch format
        rgb = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()  # depth
        depth = torch.from_numpy(depth.copy()).float()
        depth_mask = torch.from_numpy(depth_mask)
        inputs = {}
        inputs["rgb"] = rgb
        inputs["normalized_rgb"] = self.normalize(rgb.clone())
        inputs["gt_depth"] = depth
        inputs["val_mask"] = depth_mask
        # import ipdb
        # ipdb.set_trace()
        # Return the set of pano data
        return inputs


    def __len__(self):
        '''Return the size of this dataset'''
        return len(self.image_list)

    def readRGBPano(self, path):
        '''Read RGB and normalize to [0,1].'''
        rgb = cv2.imread(path)
        rgb = cv2.resize(rgb, (self.pano_w, self.pano_h), interpolation=cv2.INTER_AREA)

        return rgb

    def readDepthPano(self, path):

        return self.read_exr(path)[...,0].astype(np.float32)
        # mat_content = np.load(path, allow_pickle=True)
        # depth_img = mat_content['depth']
        # return depth_img.astype(np.float32)
        #
        # depth = cv2.imread(path, -1).astype(np.float32)
        # depth = cv2.resize(depth, (self.pano_w, self.pano_h), interpolation=cv2.INTER_AREA)
        # depth = depth / 65535 * 128
        # return depth

    def read_exr(self, image_fpath):
        f = OpenEXR.InputFile(image_fpath)
        dw = f.header()['dataWindow']
        w, h = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        im = np.empty((h, w, 3))

        # Read in the EXR
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        channels = f.channels(["R", "G", "B"], FLOAT)
        for i, channel in enumerate(channels):
            im[:, :, i] = np.reshape(array.array('f', channel), (h, w))
        return im




