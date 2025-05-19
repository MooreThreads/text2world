## Information
这一步模型分为两个部分，一个是基于 ac3d 代码的方式训练 camera-control 的cogvideoX 模型，另一个是训练 remap模型（不必须）。
## 数据预处理
数据集来源：[RealEstate10K](https://google.github.io/realestate10k/download.html) ，AC3D, DL3DV。

数据预处理的方式采用的 latentLRM/selfsplat 模型的方法进行预处理，保存场景的的图像数据和位姿数据，以及 caption 保存到 .torch 文件中。

## 模型参数
可选择 AC3D 预训练好的模型，实测效果还可以。
AC3D: CogVideoX-2B: [Checkpoint](https://drive.google.com/file/d/1RmTnF7mJ65s5TSqr4k_cthZXMWesd3nA/view)

AC3D: CogVideoX-5B: [Checkpoint](https://drive.google.com/file/d/1QsfmLmb-_Pv_pSbLrmbqBBehc9Oo6A79/view)

## 训练代码
```bash
bash scripts/train.sh
```

### Acknowledgements

- This code mainly builds upon [CogVideoX-ControlNet](https://github.com/TheDenk/cogvideox-controlnet) and [AC3D](https://github.com/snap-research/ac3d)
- This code uses the original CogVideoX model [CogVideoX](https://github.com/THUDM/CogVideo/tree/main)
