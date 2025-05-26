# Text2World
![Demo](./assets/pipline_text2world.jpg)
### Information
基于 3D高斯溅射和视频生成技术构建的 Text2world pipeline。
特性：
1. 第一个开源的基于T2V模型和 mamba-transformer架构 LRM 的 pipeline
2. 更高效的LRM 重建模型：由视频 VAE 的 encoder 和mamba-transformers结构组成，使得LRM 模型可以输入更多的图片的同时，节约更多的显存。
3. 提供两种路径构建 3DGS。Normal Path将latent 进行 decoder 和后处理得到的视频重新 经过encoder 生成 video latent;Remap Path（Experimental）: 通过 remap 模型将 video latent 直接映射到 decoder 生成的 latent 中，减少不必要的 decoder-encoder和后处理过程，为将来的端到端训练做准备。

### BENCHMARK
为了验证pipeline 的有效性以及LRM 模型的性能构建验证数据集：
* 公开数据集用于评估 LRM 在真实场景下的重建效果
* 合成数据集用于评估 LRM 在生成场景下的重建效果。我们用 llm 生成了 1000条场景提示词包含 Natural Landscapes (200 prompts)，Urban Environments (150 prompts)，Interiors (150 prompts)，Fantasy Settings (150 prompts)，Sci-Fi Settings (150 prompts)，Historical Settings (100 prompts)，Abstract Compositions (100 prompts)），并从re10k 数据集中随机抽取 camera参数作为输入。

|dataset|PSNR|SSIM|lpips|
|-----------|-----|------|-----------|
|public|29.34|0.87|0.205|
||||
### Example
|  参考示例  | 参考示例      |  参考示例   | 
|-----------|-----------|-----------|
|![Demo](./assets/demo1.gif) |![Demo](./assets/demo2.gif) |![Demo](./assets/demo3.gif) |
|![Demo](./assets/demo4.gif) | ![Demo](./assets/demo5.gif)|![Demo](./assets/demo6.gif) |
### Dataset
数据来源：
* [RealEstate10K](https://google.github.io/realestate10k/download.html)
* [DL3DV](https://dl3dv-10k.github.io/DL3DV-10K/)
* [AC3D](https://infinite-nature.github.io/)

数据处理方法，请参考
1. 位姿数据预处理： [pixelsplat](https://github.com/dcharatan/pixelsplat)
2. caption: [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun) 或者[CameraCtrl](https://github.com/hehao13/CameraCtrl).
### inference
考虑到diffusion latent与后处理视频重新 encoder的 VAE latent存在一定的差异。

现提供两种推理 pipeline。
* `nonmap_pipeline.py` 对 diffusion 模型生成 video 进行后处理后作为输入提供给 latentLRM模型进行推理生成渲染视频。
* `remap_pipeline.py` 对diffusion生成的 latent的进行 remap 从而缓解这种差异。

对应的调用代码：
- $pose_folder: 类似 RE10K 的 pose folder
- $prompt_txt: prompt列表
- $MODEL_PATH： 视频生成模型参数
- $ckpt_path： controlnet模型参数
- $lrm_weight：LRM 模型参数
- $remap_weight（可选）： remap 模型参数
- $out_dir: 输出路径

```
python generate_nonmap_api.py \
    --prompt  $prompt_txt \ 
    --lrm_weight $lrm_weight \
    --pose_folder  \
    --base_model_path $MODEL_PATH \
    --controlnet_model_path $ckpt_path \
    --output_path $out_dir \
    --start_camera_idx 0 \
    --end_camera_idx 7 \
    --stride_min 2 \
    --stride_max 2 \
    --height 480 \
    --width 720 \
    --controlnet_weights 1.0 \
    --controlnet_guidance_start 0.0 \
    --controlnet_guidance_end 0.4 \
    --controlnet_transformer_num_attn_heads 4 \
    --controlnet_transformer_attention_head_dim 64 \
    --controlnet_transformer_out_proj_dim_factor 64 \
    --num_inference_steps 20
```

```
python generate_remap_api.py \
    --prompt  $prompt_txt \ 
    --pose_folder $pose_folder \
    --lrm_weight $lrm_weight \
    --remap_weight $remap_weight \
    --base_model_path $MODEL_PATH \
    --controlnet_model_path $ckpt_path \
    --output_path $out_dir \
    --start_camera_idx 0 \
    --end_camera_idx 7 \
    --stride_min 2 \
    --stride_max 2 \
    --height 480 \
    --width 720 \
    --controlnet_weights 1.0 \
    --controlnet_guidance_start 0.0 \
    --controlnet_guidance_end 0.4 \
    --controlnet_transformer_num_attn_heads 4 \
    --controlnet_transformer_attention_head_dim 64 \
    --controlnet_transformer_out_proj_dim_factor 64 \
    --num_inference_steps 20
```

### Training 
参考`wonderland`, 将 latentLRM 模型和 video diffusion 模型分开训练。方法见次级子目录