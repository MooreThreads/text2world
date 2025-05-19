import sys
from pathlib import Path
from controlnet_lrm_pipeline5 import ControlnetCogVideoXPipeline, LatentAlign3D
from cogvideo_transformer import CustomCogVideoXTransformer3DModel
from cogvideo_controlnet import CogVideoXControlnet
from training.controlnet_datasets_camera import ray_condition, Camera
from torchvision.transforms.functional import to_pil_image
from datetime import datetime
import argparse
import os
import numpy as np
import torch
import cv2
from io import BytesIO
from PIL import Image
import random
from transformers import T5EncoderModel, T5Tokenizer
from latent_lrm import LongLRM
from diffusers import (
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    AutoencoderKLCogVideoX
)
from diffusers.utils import export_to_video, load_video
from controlnet_aux import HEDdetector, CannyDetector

from inference.utils import stack_images_horizontally


def load_video(video_path):
    """加载视频帧并返回JPEG字节流的字典,键为帧索引"""
    cap = cv2.VideoCapture(str(video_path))
    frames = {}
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 转换为RGB格式
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 转换为PIL Image并保存为JPEG字节流
        img = Image.fromarray(frame)
        byte_stream = BytesIO()
        img.save(byte_stream, format='JPEG')
        jpeg_bytes = byte_stream.getvalue()

        # 保存为torch张量
        frames[frame_idx] = torch.tensor(
            np.frombuffer(jpeg_bytes, dtype=np.uint8))
        frame_idx += 1

    cap.release()
    return frames


class VideoGenerator:
    def __init__(self, args):
        self.args = args
        self.dtype = torch.bfloat16
        self.device = "cuda"
        lrm = LongLRM(version=4)
        weight_path = '/jfs/qian.ren/latentLRM/checkpoints/7m1tx3+cogvideox+bs32+dl3dv+re10k+acid+low+dl3dv-high+ver-4/checkpoint_000041400.pt'
        state_dict = torch.load(weight_path, map_location=self.device)['model']
        state_dict = {k.replace('module.', ''): v for k,
                      v in state_dict.items()}
        lrm.load_state_dict(state_dict, strict=False)
        self.lrm = lrm
        weight_path = '/jfs/qian.ren/latentLRM/ac3d/test/CogVideoX-5b_covideoX-5B-ac3d_10000/checkpoints/best_model.pth'
        state_dict = torch.load(weight_path, map_location=self.device)
        latent_mapper = LatentAlign3D(16)
        latent_mapper.load_state_dict(state_dict, strict=False)
        self.latent_mapper = latent_mapper
        # Initialize components
        self.tokenizer = T5Tokenizer.from_pretrained(
            args.base_model_path, subfolder="tokenizer"
        )
        self.text_encoder = T5EncoderModel.from_pretrained(
            args.base_model_path, subfolder="text_encoder"
        )
        self.transformer = CustomCogVideoXTransformer3DModel.from_pretrained(
            args.base_model_path, subfolder="transformer"
        )
        self.vae = AutoencoderKLCogVideoX.from_pretrained(
            args.base_model_path, subfolder="vae"
        )
        self.scheduler = CogVideoXDDIMScheduler.from_pretrained(
            args.base_model_path, subfolder="scheduler"
        )

        # ControlNet setup
        num_attention_heads_orig = 48 if "5b" in args.base_model_path.lower() else 30
        controlnet_kwargs = {}
        if args.controlnet_transformer_num_attn_heads is not None:
            controlnet_kwargs["num_attention_heads"] = args.controlnet_transformer_num_attn_heads
        else:
            controlnet_kwargs["num_attention_heads"] = num_attention_heads_orig
        if args.controlnet_transformer_attention_head_dim is not None:
            controlnet_kwargs["attention_head_dim"] = args.controlnet_transformer_attention_head_dim
        if args.controlnet_transformer_out_proj_dim_factor is not None:
            controlnet_kwargs["out_proj_dim"] = num_attention_heads_orig * \
                args.controlnet_transformer_out_proj_dim_factor
        controlnet_kwargs["out_proj_dim_zero_init"] = args.controlnet_transformer_out_proj_dim_zero_init

        self.controlnet = CogVideoXControlnet(
            num_layers=8,
            downscale_coef=8,
            in_channels=6, **controlnet_kwargs,
        )

        if args.controlnet_model_path:
            ckpt = torch.load(args.controlnet_model_path,
                              map_location='cpu', weights_only=False)
            controlnet_state_dict = {}
            for name, params in ckpt['state_dict'].items():
                controlnet_state_dict[name] = params
            m, u = self.controlnet.load_state_dict(
                controlnet_state_dict, strict=False)
            print(
                f'[ Weights from pretrained controlnet was loaded into controlnet ] [M: {len(m)} | U: {len(u)}]')

        # Full pipeline
        self.pipe = ControlnetCogVideoXPipeline(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            transformer=self.transformer,
            vae=self.vae,
            controlnet=self.controlnet,
            scheduler=self.scheduler,
            lrm=self.lrm,
            latent_mapper=self.latent_mapper
        )

        # LoRA setup if provided
        if args.lora_path:
            self.pipe.load_lora_weights(
                args.lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
            self.pipe.fuse_lora(lora_scale=1 / args.lora_rank)

        # Set scheduler
        self.pipe.scheduler = CogVideoXDPMScheduler.from_config(
            self.pipe.scheduler.config, timestep_spacing="trailing")

        # Move to device and enable optimizations
        self.pipe = self.pipe.to(dtype=self.dtype)

        self.pipe = self.pipe.to(self.device)

        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()

        # Load pose files
        self.anno_folder = '/jfs/qian.ren/latentLRM/data/dataset/re10k/annotation/train'
        self.pose_files = []
        for file in os.listdir(self.anno_folder):
            if file.endswith('.txt'):
                self.pose_files.append(os.path.join(self.anno_folder, file))

        if not self.pose_files:
            raise ValueError("No pose files found in the specified directory")

    def load_random_pose(self, args):
        """Randomly select and load a pose file"""
        cam_params = []
        while len(cam_params) < 49:
            pose_file = random.choice(self.pose_files)
            with open(pose_file, 'r') as f:
                poses = f.readlines()
            poses = [pose.strip().split(' ') for pose in poses[1:]]
            cam_params = [[float(x) for x in pose] for pose in poses]
            cam_params = [Camera(cam_param) for cam_param in cam_params]
            cam_params = cam_params[:49]  # Take first 49 frames for testing
        poses = [cam_param.params for cam_param in cam_params]

        intrinsics = np.asarray([[cam_param.fx * args.width,
                                cam_param.fy * args.height,
                                cam_param.cx * args.width,
                                cam_param.cy * args.height]
                                for cam_param in cam_params], dtype=np.float32)
        intrinsics = torch.as_tensor(intrinsics)[None]  # [1, n_frame, 4]

        c2w_poses = np.array(
            [cam_param.c2w_mat for cam_param in cam_params], dtype=np.float32)
        c2w = self.normalize_pose(torch.as_tensor(c2w_poses))
        c2w = c2w[None]  # [1, n_frame, 4, 4]
        controlnet_latents = ray_condition(
            intrinsics, c2w, args.height, args.width, device='cpu')[0]
        controlnet_latents = controlnet_latents.permute(
            0, 3, 1, 2).contiguous()[None].to(self.device)

        return controlnet_latents, c2w, intrinsics, poses

    def normalize_pose(self, c2w):
        import torch.nn.functional as F
        # noramlize input camera poses
        position_avg = c2w[:, :3, 3].mean(0)  # (3,)
        forward_avg = c2w[:, :3, 2].mean(0)  # (3,)
        down_avg = c2w[:, :3, 1].mean(0)  # (3,)
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

        c2w = torch.matmul(pos_avg_inv.unsqueeze(0), c2w)

        # scale scene size
        position_max = c2w[:, :3, 3].abs().max()
        scene_scale = 1.0 * position_max
        scene_scale = 1.0 / scene_scale
        c2w[:, :3, 3] *= scene_scale
        return c2w

    def load_random_prompt(self, prompt_files):
        prompts = []
        with open(prompt_files, 'r') as f:
            for prompt in f.readlines():
                prompts.append(prompt.strip())
        return random.choice(prompts)

    def generate_video(self, output_path=None):
        """Generate video with random pose"""
        if output_path is None:
            output_path = self.args.output_path
        os.makedirs(output_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d%H")
        dataset_path = os.path.join(
            '/jfs/qian.ren/latentLRM/data/dataset', f'generate-{timestamp}', 'test')
        os.makedirs(dataset_path, exist_ok=True)
        for seed in range(self.args.num_videos_per_prompt):
            prefill = len(os.listdir(output_path))
            controlnet_latents, c2ws, intrinsics, pose = self.load_random_pose(
                args)
            c2ws = c2ws.to(self.device)
            intrinsics = intrinsics.to(self.device)
            video, nvideo, rvideo, renders = self.pipe(
                prompt=self.load_random_prompt(self.args.prompt),
                height=self.args.height,
                width=self.args.width,
                c2ws=c2ws,
                intrinsics=intrinsics,
                controlnet_latents=controlnet_latents,
                num_videos_per_prompt=1,
                num_inference_steps=self.args.num_inference_steps,
                num_frames=self.args.num_frames,
                use_dynamic_cfg=self.args.use_dynamic_cfg,
                guidance_scale=self.args.guidance_scale,
                generator=torch.Generator().manual_seed(seed),
                controlnet_weights=self.args.controlnet_weights,
                controlnet_guidance_start=self.args.controlnet_guidance_start,
                controlnet_guidance_end=self.args.controlnet_guidance_end,
            )

            # video_generate = video_generate_all[0]
            output_path_file = os.path.join(
                output_path, f"video_{prefill:06d}.mp4")
            export_to_video(nvideo[0], output_path_file, fps=8)
            output_path_file = os.path.join(
                output_path, f"pp_video_{prefill:06d}.mp4")
            export_to_video(video.permute(0, 2, 3, 4, 1).contiguous()[
                            0].float().cpu().numpy(), output_path_file, fps=8)
            output_path_file = os.path.join(
                output_path, f"remap_video_{prefill:06d}.mp4")
            export_to_video(rvideo.permute(0, 2, 3, 4, 1).contiguous()[
                            0].float().cpu().numpy(), output_path_file, fps=8)
            pose = torch.Tensor(pose)
            chunk = {}
            chunk['dlatent'] = dlatent
            chunk['vlatent'] = vlatent
            torch_path = f'{output_path}/{prefill:06d}.torch'
            torch.save(chunk, torch_path)

            print(f'{output_path_file}\t{torch_path}')
            output_path_file = os.path.join(
                output_path, f"render_{prefill:06d}.mp4")
            renders = renders.permute(
                0, 2, 3, 4, 1).contiguous()  # (B,  V, C, H, W)
            export_to_video(renders[0].cpu().numpy(), output_path_file, fps=8)
            # pose = torch.Tensor(pose)
            # images = load_video(output_path_file)
            # chunk ={}
            # chunk['images'] = images
            # chunk['cameras'] = pose
            # chunk['key'] = f'{prefill:06d}'
            # torch_path = f'{dataset_path}/{prefill:06d}.torch'
            # torch.save(chunk,torch_path)
            # print(f'{output_path_file}\t{torch_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--prompt", type=str, required=True,
                        help="The description of the video to be generated")
    parser.add_argument(
        "--lrm_weight", type=str, default="/jfs/qian.ren/latentLRM/checkpoints/7m1tx3+cogvideox+bs32+dl3dv+re10k+acid+low+dl3dv-high+ver-4/checkpoint_000041400.pt", help="The path of the pre-trained model to be used"
    )
    parser.add_argument(
        "--pose_folder", type=str, default="/jfs/qian.ren/latentLRM/data/dataset/re10k/annotation/train"
    )
    parser.add_argument(
        "--remap_weight", type=str, default="/jfs/qian.ren/latentLRM/checkpoints/7m1tx3+cogvideox+bs32+dl3dv+re10k+acid+low+dl3dv-high+ver-4/checkpoint_000041400.pt", help="The path of the pre-trained model to be used"
    )
    parser.add_argument(
        "--base_model_path", type=str, default="THUDM/CogVideoX-5b", help="The path of the pre-trained model to be used"
    )
    parser.add_argument(
        "--controlnet_model_path", type=str, default="TheDenk/cogvideox-5b-controlnet-hed-v1", help="The path of the controlnet pre-trained model to be used"
    )
    parser.add_argument("--pose_file", type=str, default='pose.txt',
                        help="pose_file (deprecated, will use random pose)")
    parser.add_argument("--controlnet_weights", type=float,
                        default=0.5, help="Strenght of controlnet")
    parser.add_argument("--controlnet_guidance_start", type=float, default=0.0,
                        help="The stage when the controlnet starts to be applied")
    parser.add_argument("--controlnet_guidance_end", type=float, default=0.5,
                        help="The stage when the controlnet end to be applied")
    parser.add_argument("--use_dynamic_cfg", type=bool,
                        default=True, help="Use dynamic cfg")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="The path of the LoRA weights to be used")
    parser.add_argument("--lora_rank", type=int, default=128,
                        help="The rank of the LoRA weights")
    parser.add_argument(
        "--output_path", type=str, default="./output", help="The path where the generated video will be saved"
    )
    parser.add_argument("--guidance_scale", type=float, default=6.0,
                        help="The scale for classifier-free guidance")
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of steps for the inference process"
    )
    parser.add_argument("--num_videos_per_prompt", type=int,
                        default=100, help="Number of videos to generate per prompt")
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="The data type for computation (e.g., 'float16' or 'bfloat16')"
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="The seed for reproducibility")
    parser.add_argument("--stride_min", type=int, default=1)
    parser.add_argument("--stride_max", type=int, default=1)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--start_camera_idx", type=int, default=0)
    parser.add_argument("--end_camera_idx", type=int, default=1)
    parser.add_argument(
        "--controlnet_transformer_num_attn_heads", type=int, default=None)
    parser.add_argument(
        "--controlnet_transformer_attention_head_dim", type=int, default=None)
    parser.add_argument(
        "--controlnet_transformer_out_proj_dim_factor", type=int, default=None)
    parser.add_argument("--controlnet_transformer_out_proj_dim_zero_init", action="store_true", default=False, help=("Init project zero."),
                        )
    parser.add_argument("--downscale_coef", type=int, default=8)
    parser.add_argument("--controlnet_input_channels", type=int, default=6)

    args = parser.parse_args()

    # Initialize generator with loaded models
    generator = VideoGenerator(args)

    # Generate video with random pose
    generator.generate_video()
