import os
import random
import json
import torch
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
from io import BytesIO
import PIL.Image as Image
from pathlib import Path
# from decord import VideoReader
from torch.utils.data.dataset import Dataset
from packaging import version as pver
import traceback


def unpack_mm_params(p):
    """
    Unpack parameters that could be either a tuple/list or single value.

    Args:
        p: Input parameter that could be either:
           - Tuple/list: returns the first two elements
           - Single number: returns the same value twice
           - Other: raises Exception

    Returns:
        Tuple of unpacked parameters

    Raises:
        Exception: If input type is not tuple/list or number
    """
    if isinstance(p, (tuple, list)):
        return p[0], p[1]
    elif isinstance(p, (int, float)):
        return p, p
    raise Exception(
        f'Unknown input parameter type.\nParameter: {p}.\nType: {type(p)}')


class RandomHorizontalFlipWithPose(nn.Module):
    """
    Random horizontal flip augmentation that maintains consistency with pose information.
    Can apply different flip operations to different images in a batch.
    """

    def __init__(self, p=0.5):
        """
        Initialize the flip probability.

        Args:
            p: Probability of flipping an image (default: 0.5)
        """
        super(RandomHorizontalFlipWithPose, self).__init__()
        self.p = p

    def get_flip_flag(self, n_image):
        """
        Generate random flip flags for a batch of images.

        Args:
            n_image: Number of images in the batch

        Returns:
            Boolean tensor indicating which images to flip
        """
        return torch.rand(n_image) < self.p

    def forward(self, image, flip_flag=None):
        """
        Apply horizontal flip to images based on flip flags.

        Args:
            image: Input image tensor [N, C, H, W]
            flip_flag: Optional pre-computed flip flags. If None, generates new ones.

        Returns:
            Flipped or original images based on flags
        """
        n_image = image.shape[0]
        if flip_flag is not None:
            assert n_image == flip_flag.shape[0]
        else:
            flip_flag = self.get_flip_flag(n_image)

        ret_images = []
        for fflag, img in zip(flip_flag, image):
            if fflag:
                ret_images.append(F.hflip(img))
            else:
                ret_images.append(img)
        return torch.stack(ret_images, dim=0)


class Camera(object):
    """
    Camera parameter container and transformation utility class.
    Stores intrinsic and extrinsic camera parameters.
    """

    def __init__(self, entry):
        """
        Initialize camera parameters from input array.

        Args:
            entry: Array containing [fx, fy, cx, cy, ...] followed by world-to-camera matrix
        """
        fx, fy, cx, cy = entry[:4]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[6:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)
        self.params = entry[1:]


def custom_meshgrid(*args):
    """
    Create meshgrid with consistent behavior across PyTorch versions.

    Args:
        *args: Input tensors to create grid from

    Returns:
        Meshgrid tensor
    """
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def ray_condition(K, c2w, H, W, device, flip_flag=None):
    """
    Compute Plücker coordinates for rays through each pixel in camera images.

    Args:
        K: Camera intrinsics tensor [B, V, 4] (fx, fy, cx, cy)
        c2w: Camera-to-world matrices [B, V, 4, 4]
        H: Image height
        W: Image width
        device: Device to compute on
        flip_flag: Optional flags indicating which views to flip

    Returns:
        Plücker coordinates tensor [B, V, H, W, 6]
    """
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B, V = K.shape[:2]

    # Create pixel coordinate grids
    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + \
        0.5          # [B, V, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + \
        0.5          # [B, V, HxW]

    # Handle flipped views if needed
    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0:
        j_flip, i_flip = custom_meshgrid(
            torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
            torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype)
        )
        i_flip = i_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        j_flip = j_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip

    # Compute ray directions in camera space
    fx, fy, cx, cy = K.chunk(4, dim=-1)     # B,V, 1

    zs = torch.ones_like(i)                 # [B, V, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)              # B, V, HW, 3
    directions = directions / \
        directions.norm(dim=-1, keepdim=True)             # B, V, HW, 3

    # Transform directions to world space
    rays_d = directions @ c2w[..., :3,
                              :3].transpose(-1, -2)        # B, V, HW, 3
    rays_o = c2w[..., :3, 3]                                        # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(
        rays_d)                   # B, V, HW, 3

    # Compute Plücker coordinates (moment and direction)
    # B, V, HW, 3
    rays_dxo = torch.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(
        B, c2w.shape[1], H, W, 6)             # B, V, H, W, 6
    return plucker


class RealEstate10KPoseControlnetDataset(Dataset):
    """
    Dataset class for RealEstate10K dataset with pose information.
    Provides video frames, camera poses, and generates Plücker coordinates for control.
    """

    def __init__(
            self,
            video_root_dir,
            annotation_json=None,
            stage='train',
            stride=(1, 1),
            sample_n_frames=49,
            relative_pose=True,
            zero_t_first_frame=True,
            image_size=[480, 720],
            rescale_fxy=True,
            shuffle_frames=False,
            hflip_p=0.0,
            datasets=['dl3dv', 're10k']
    ):
        """
        Initialize dataset parameters and transformations.

        Args:
            video_root_dir: Root directory containing videos
            annotation_json: JSON file with video metadata
            stride: Tuple of (minimum stride, sample stride) for frame sampling
            sample_n_frames: Number of frames to sample per video
            relative_pose: Whether to use relative camera poses
            zero_t_first_frame: Whether to set first frame translation to zero
            image_size: Target image size [H, W]
            rescale_fxy: Whether to rescale focal lengths when resizing
            shuffle_frames: Whether to shuffle frame order
            hflip_p: Probability of horizontal flip augmentation
        """
        # minimum_sample_stride, sample_stride = stride
        if hflip_p != 0.0:
            use_flip = True
        else:
            use_flip = False
        root_path = video_root_dir
        self.root_path = Path(root_path)
        self.stage = stage
        self.datasets = datasets
        self.relative_pose = relative_pose
        self.zero_t_first_frame = zero_t_first_frame
        # self.sample_stride = sample_stride
        # self.minimum_sample_stride = minimum_sample_stride
        self.sample_n_frames = sample_n_frames

        # Load dataset metadata
        self.chunks = []
        for dataset in datasets:
            data_path = self.root_path / dataset / self.stage
            data_chunks = sorted(
                [path for path in data_path.iterdir() if path.suffix == ".torch"]
            )
            self.chunks.extend(data_chunks)
        self.length = len(self.chunks)

        # Set up image transforms
        sample_size = image_size
        sample_size = tuple(sample_size) if not isinstance(
            sample_size, int) else (sample_size, sample_size)
        self.sample_size = sample_size
        if use_flip:
            pixel_transforms = [transforms.Resize(sample_size),
                                RandomHorizontalFlipWithPose(hflip_p),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)]
        else:
            pixel_transforms = [transforms.Resize(sample_size),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)]
        self.rescale_fxy = rescale_fxy
        self.sample_wh_ratio = sample_size[1] / sample_size[0]

        self.pixel_transforms = pixel_transforms
        self.shuffle_frames = shuffle_frames
        self.use_flip = use_flip

    def get_relative_pose(self, cam_params):
        """
        Convert absolute camera poses to relative poses w.r.t. first frame.

        Args:
            cam_params: List of Camera objects

        Returns:
            Array of relative camera poses [n_frames, 4, 4]
        """
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        source_cam_c2w = abs_c2ws[0]

        # Compute camera-to-origin distance for first frame
        if self.zero_t_first_frame:
            cam_to_origin = 0
        else:
            cam_to_origin = np.linalg.norm(source_cam_c2w[:3, 3])

        # Create target camera pose (looking down negative z-axis)
        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -cam_to_origin],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Compute relative poses
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [target_cam_c2w, ] + \
            [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses

    def load_cameras(self, chunk):
        """
        Load camera parameters for given index.

        Args:
            idx: Dataset index

        Returns:
            List of Camera objects for each frame
        """

        cam_params = chunk['cameras'].numpy()
        cam_params = [Camera(cam_param) for cam_param in cam_params]
        return cam_params

    def get_batch(self, idx):
        """
        Get a batch of data for training.

        Args:
            idx: Dataset index

        Returns:
            Tuple containing:
            - pixel_values: Video frames [n_frames, C, H, W]
            - video_caption: Text caption
            - plucker_embedding: Plücker coordinates [n_frames, 6, H, W]
            - flip_flag: Flags indicating flipped frames
            - clip_name: Video clip name
        """
        # Load video and camera data
        chunk_path = Path(self.chunks[idx])
        # read config
        if 'dl3dv' in str(chunk_path):
            chunk = torch.load(chunk_path)[0]
            minimum_sample_stride = 1
            sample_stride = 2
        elif 're10k' in str(chunk_path):
            chunk = torch.load(chunk_path)
            minimum_sample_stride = 3
            sample_stride = 5
        clip_name = chunk['key']
        if 'dl3dv_' in clip_name:
            clip_name = clip_name.lstrip('dl3dv_')
        cam_params = self.load_cameras(chunk)
        assert len(cam_params) >= self.sample_n_frames
        total_frames = len(cam_params)
        video_caption = open(chunk_path.parent.parent /
                             'clip' / f'{clip_name}.txt', 'r').read()
        # Determine sampling stride
        current_sample_stride = sample_stride
        if total_frames < self.sample_n_frames * current_sample_stride:
            maximum_sample_stride = int(total_frames // self.sample_n_frames)
            current_sample_stride = random.randint(
                minimum_sample_stride, maximum_sample_stride)

        # Sample frame indices
        cropped_length = self.sample_n_frames * current_sample_stride
        start_frame_ind = random.randint(
            0, max(0, total_frames - cropped_length - 1))
        end_frame_ind = min(start_frame_ind + cropped_length, total_frames)

        assert end_frame_ind - start_frame_ind >= self.sample_n_frames
        frame_indices = np.linspace(
            start_frame_ind, end_frame_ind - 1, self.sample_n_frames, dtype=int)

        # Optionally shuffle frames
        if self.shuffle_frames:
            perm = np.random.permutation(self.sample_n_frames)
            frame_indices = frame_indices[perm]

        # Load and normalize frames
        images = [Image.open(BytesIO(chunk['images'][frame].numpy().tobytes()))
                  for frame in frame_indices]
        images = np.stack([np.array(image)
                          for image in images])  # (num_frames, H, W, 3)
        pixel_values = torch.from_numpy(
            images).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.

        # Process camera parameters
        cam_params = [cam_params[indice] for indice in frame_indices]
        if self.rescale_fxy:
            # Adjust focal lengths to account for image resizing
            ori_h, ori_w = pixel_values.shape[-2:]
            ori_wh_ratio = ori_w / ori_h
            if ori_wh_ratio > self.sample_wh_ratio:       # rescale fx
                resized_ori_w = self.sample_size[0] * ori_wh_ratio
                for cam_param in cam_params:
                    cam_param.fx = resized_ori_w * \
                        cam_param.fx / self.sample_size[1]
            else:                                          # rescale fy
                resized_ori_h = self.sample_size[1] / ori_wh_ratio
                for cam_param in cam_params:
                    cam_param.fy = resized_ori_h * \
                        cam_param.fy / self.sample_size[0]

        # Prepare camera intrinsics tensor
        intrinsics = np.asarray([[cam_param.fx * self.sample_size[1],
                                  cam_param.fy * self.sample_size[0],
                                  cam_param.cx * self.sample_size[1],
                                  cam_param.cy * self.sample_size[0]]
                                 for cam_param in cam_params], dtype=np.float32)
        intrinsics = torch.as_tensor(
            intrinsics)[None]                  # [1, n_frame, 4]

        # Get camera poses (relative or absolute)
        if self.relative_pose:
            c2w_poses = self.get_relative_pose(cam_params)
        else:
            c2w_poses = np.array(
                [cam_param.c2w_mat for cam_param in cam_params], dtype=np.float32)
        # [1, n_frame, 4, 4]
        c2w = torch.as_tensor(c2w_poses)[None]

        # Generate flip flags if needed
        if self.use_flip:
            flip_flag = self.pixel_transforms[1].get_flip_flag(
                self.sample_n_frames)
        else:
            flip_flag = torch.zeros(
                self.sample_n_frames, dtype=torch.bool, device=c2w.device)

        # Compute Plücker coordinates for control
        plucker_embedding = ray_condition(intrinsics, c2w, self.sample_size[0], self.sample_size[1], device='cpu',
                                          flip_flag=flip_flag)[0].permute(0, 3, 1, 2).contiguous()
        return pixel_values, video_caption, plucker_embedding, flip_flag, clip_name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                video, video_caption, plucker_embedding, flip_flag, clip_name = self.get_batch(
                    idx)
                break

            except Exception as e:
                # traceback.print_exc()
                idx = random.randint(0, self.length - 1)
        if self.use_flip:
            video = self.pixel_transforms[0](video)
            video = self.pixel_transforms[1](video, flip_flag)
            video = self.pixel_transforms[2](video)
        else:
            for transform in self.pixel_transforms:
                video = transform(video)
        data = {
            'video': video,
            'caption': video_caption,
            'controlnet_video': plucker_embedding,
        }
        return data


if __name__ == '__main__':
    dataset = RealEstate10KPoseControlnetDataset(
        '/jfs/qian.ren/latentLRM/data/dataset', datasets=['dl3dv'])
    for i in range(len(dataset)):
        print(dataset[i].keys())
        print(dataset[i]['video'].shape)
        print(dataset[i]['controlnet_video'].shape)
        print(dataset[i]['caption'])
        break
