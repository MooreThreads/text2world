import os
from io import BytesIO
import json
import random
import traceback
import numpy as np
import PIL.Image as Image
from einops import rearrange, repeat
import cv2
import torch
from pathlib import Path
from jaxtyping import Float, UInt8
from torch import Tensor
import torchvision.transforms as tf
import torch.nn.functional as F
from easydict import EasyDict as edict
from torch.utils.data import Dataset
import torch.nn as nn
# 该版本适应selfsplate数据集处理方式


class Dataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.evaluation = config.get("evaluation", False)
        if self.evaluation and "data_eval" in config:
            self.config.data.update(config.data_eval)
        self.stage = 'test' if self.evaluation else 'train'
        data_root = Path('/jfs/qian.ren/latentLRM/data/dataset')
        datasets = self.config.data.datasets
        self.to_tensor = tf.ToTensor()
        self.chunks = []
        for dataset in datasets:
            data_path = data_root / dataset / self.stage
            data_chunks = sorted(
                [path for path in data_path.iterdir() if path.suffix == ".torch"]
            )
            self.chunks.extend(data_chunks)

    def __len__(self):
        return len(self.chunks)

    def process_frames(self, frames_idx, chunk):
        # print(frames_idx)
        resize_h = self.config.data.get("resize_h", -1)
        resize_w = self.config.data.get("resize_w", -1)
        # patch_size = self.config.model.patch_size
        # patch_size = patch_size * 2 ** len(self.config.model.get("merge_layers", []))
        square_crop = self.config.data.square_crop
        images = [Image.open(
            BytesIO(chunk['images'][int(frame)].numpy().tobytes())) for frame in frames_idx]

        poses = chunk["cameras"][frames_idx]

        images = np.stack([np.array(image)
                          for image in images])  # (num_frames, H, W, 3)
        h, w = images.shape[1:3]
        if images.shape[1] > images.shape[2]:
            raise ValueError("Image height must be less than width")

        if resize_h == -1 and resize_w == -1:
            resize_h = images.shape[1]
            resize_w = images.shape[2]
        elif resize_h == -1:
            resize_h = int(resize_w / images.shape[2] * images.shape[1])
        elif resize_w == -1:
            resize_w = int(resize_h / images.shape[1] * images.shape[2])
        images = np.stack([cv2.resize(image, (resize_w, resize_h))
                          for image in images])
        if square_crop:
            min_size = min(resize_h, resize_w)
            # center crop
            start_h = (resize_h - min_size) // 2
            start_w = (resize_w - min_size) // 2
            images = images[:, start_h:start_h +
                            min_size, start_w:start_w+min_size, :]
        images = torch.from_numpy(images).permute(
            0, 3, 1, 2).float()  # (num_frames, 3, resize_h, resize_w)
        images = images / 255.0
        c2ws, intrinsics = self.convert_poses(poses)
        intrinsics[:, 0] *= resize_w
        intrinsics[:, 1] *= resize_h
        intrinsics[:, 2] *= resize_w
        intrinsics[:, 3] *= resize_h
        if square_crop:
            intrinsics[:, 2] -= start_w
            intrinsics[:, 3] -= start_h
        return images, intrinsics, c2ws

    def __getitem__(self, idx):
        chunk_path = self.chunks[idx]
        if 'dl3dv' in str(chunk_path):
            per_frame_dist = np.random.randint(1, 2)
            chunk = torch.load(chunk_path)[0]
        elif 're10k' in str(chunk_path):
            per_frame_dist = np.random.randint(3, 5)
            chunk = torch.load(chunk_path)
        elif 'acid' in str(chunk_path):
            per_frame_dist = np.random.randint(1, 2)
            chunk = torch.load(chunk_path)
        if 'generate' in str(chunk_path):
            # dataset = chunk_path.split('/')[-3]
            per_frame_dist = np.random.randint(1)
            chunk = torch.load(chunk_path)
            frames = chunk['images']
            scene_name = chunk['key']
            target_frame_idx = torch.arange(49)
            input_images, input_intr, target_c2ws = self.process_frames(
                target_frame_idx, chunk)
            input_c2ws, target_c2ws, pos_avg_inv, scene_scale = self.normalize_pose(
                target_c2ws, target_c2ws)
            num_frames, C, H, W = input_images.shape
            torch.inverse(input_c2ws)
            torch.inverse(target_c2ws)
            ret_dict = {
                "scene_name": scene_name,
                "input_images": input_images,
                "input_intr": input_intr,
                "input_c2ws": input_c2ws,
                "test_images": input_images,
                "test_intr": input_intr,
                "test_c2ws": input_c2ws,
                "pos_avg_inv": pos_avg_inv,
                "scene_scale": scene_scale,
                # "dataset": dataset
                # "target_frame_idx":sorted(target_frame_idx),
                # "input_frame_idx":input_frame_idx,
            }
            return ret_dict
        try:
            frames = chunk['images']
            scene_name = chunk['key']
            # per_frame_dist = np.random.randint(1,2)
            num_input_frames = self.config.data.num_input_frames
            num_target_frames = self.config.data.get("num_target_frames")
            if per_frame_dist > 1:
                frame_dist = min(per_frame_dist * 49, len(frames))
            else:
                frame_dist = num_input_frames+num_target_frames

            if num_input_frames + num_target_frames > len(frames) and self.stage != 'test':
                raise ValueError(
                    f'Skip this scene for short,Number of frame:{len(frames)},Scene_name:{scene_name}')
            # get frame rangez
            if len(frames) - frame_dist == 0:
                start_frame_idx = 0
            else:
                start_frame_idx = np.random.randint(len(frames) - frame_dist)
            end_frame_idx = start_frame_idx + frame_dist
            input_frame_idx = torch.linspace(
                start_frame_idx,
                end_frame_idx-1,
                num_input_frames,
                dtype=torch.int64,
            )
            # 创建所有可能的帧索引
            all_indices = torch.arange(
                start_frame_idx, end_frame_idx, dtype=torch.int64)
            # 移除已用于context的索引
            available_indices = torch.tensor(
                [idx for idx in all_indices if idx not in input_frame_idx])
            # 从剩余索引中随机选择target views

            target_frame_idx_unseen = available_indices[torch.randperm(
                len(available_indices))[:num_target_frames]]
            target_frame_idx_seen = input_frame_idx[torch.randperm(len(input_frame_idx))[
                :num_target_frames]]
            target_frame_idx = torch.cat(
                [target_frame_idx_unseen, target_frame_idx_seen])
            # 随机方向取反
            reverse_input_prob = self.config.data.get(
                "reverse_input_prob", 0.0)
            reverse_input = np.random.rand() < reverse_input_prob
            if reverse_input:
                reversed_input_frame_idx = torch.flip(
                    input_frame_idx, dims=[0])
            # target_frames = [frames[i] for i in target_frame_idx]
            target_images, target_intr, target_c2ws = self.process_frames(
                target_frame_idx, chunk)

            # input_frames = [frames[i] for i in input_frame_idx]
            input_images, input_intr, input_c2ws = self.process_frames(
                input_frame_idx, chunk)

            input_c2ws, target_c2ws, pos_avg_inv, scene_scale = self.normalize_pose(
                input_c2ws, target_c2ws)
            # pos_avg_inv = None
            # scene_scale = None
            _, C, H, W = target_images.shape
            # target_images = nn.functional.interpolate(target_images, (H//2, W//2))
            # target_intr[:, 0] *= 0.5
            # target_intr[:, 1] *= 0.5
            # target_intr[:, 2] *= 0.5
            # target_intr[:, 3] *= 0.5
            # try inverse pose
            torch.inverse(input_c2ws)
            torch.inverse(target_c2ws)
            ret_dict = {
                "scene_name": scene_name,
                "input_images": input_images,
                "input_intr": input_intr,
                "input_c2ws": input_c2ws,
                "test_images": target_images,
                "test_intr": target_intr,
                "test_c2ws": target_c2ws,
                "pos_avg_inv": pos_avg_inv,
                "scene_scale": scene_scale,
                # "target_frame_idx":sorted(target_frame_idx),
                # "input_frame_idx":input_frame_idx,
            }
        except:
            # traceback.print_exc()
            # print(f"error loading")
            return self.__getitem__(random.randint(0, len(self) - 1))

        return ret_dict

    def normalize_pose(self, input_c2ws, target_c2ws):
        # noramlize input camera poses
        position_avg = input_c2ws[:, :3, 3].mean(0)  # (3,)
        forward_avg = input_c2ws[:, :3, 2].mean(0)  # (3,)
        down_avg = input_c2ws[:, :3, 1].mean(0)  # (3,)
        # gram-schmidt process
        forward_avg = F.normalize(forward_avg, dim=0)
        down_avg = F.normalize(
            down_avg - down_avg.dot(forward_avg) * forward_avg, dim=0)
        right_avg = torch.cross(down_avg, forward_avg)
        pos_avg = torch.stack(
            [right_avg, down_avg, forward_avg, position_avg], dim=1)  # (3, 4)
        pos_avg = torch.cat([pos_avg, torch.tensor(
            [[0, 0, 0, 1]], device=pos_avg.device).float()], dim=0)  # (4, 4)
        pos_avg_inv = torch.inverse(pos_avg)

        input_c2ws = torch.matmul(pos_avg_inv.unsqueeze(0), input_c2ws)
        target_c2ws = torch.matmul(pos_avg_inv.unsqueeze(0), target_c2ws)

        # scale scene size
        position_max = input_c2ws[:, :3, 3].abs().max()
        scene_scale = self.config.data.get("scene_scale", 1.0) * position_max
        scene_scale = 1.0 / scene_scale

        input_c2ws[:, :3, 3] *= scene_scale
        target_c2ws[:, :3, 3] *= scene_scale
        return input_c2ws, target_c2ws, pos_avg_inv, scene_scale

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 4"],  # intrinsics
    ]:
        b, _ = poses.shape
        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = poses[:, :4]
        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32),
                     "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics


if __name__ == "__main__":
    # test dataset
    config = edict()
    config.evaluation = True
    config.data = edict()
    config.model = edict()
    config.data.resize_h = 240
    config.data.resize_w = 360
    config.data.square_crop = False
    config.data.datasets = ['generate']
    config.data.input_frame_select_type = "uniform"
    config.data.target_frame_select_type = "uniform"
    config.data.num_input_frames = 49
    config.data.num_target_frames = 24
    dataset = Dataset(config)
    print("dataset length:", len(dataset))

    for i in range(len(dataset)):
        data = dataset[i]
        print("scene_name:", data["scene_name"])
        print("input_images:", data["input_images"].shape)
        print("input_intr:", data["input_intr"].shape)
        print("input_c2ws:", data["input_c2ws"].shape)
        print("target_images:", data["test_images"].shape)
        print("target_intr:", data["test_intr"].shape)
        print("target_c2ws:", data["test_c2ws"].shape)
        print("pos_avg_inv:", data["pos_avg_inv"].shape)
        print("scene_scale:", data["scene_scale"])
        # print("target_frame_idx:", data["target_frame_idx"])
        # print("input_frame_idx:", data["input_frame_idx"])
        break
