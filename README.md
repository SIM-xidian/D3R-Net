# D3R-Net: Denoising Diffusion-based Defense Restore Network for Adversarial Defense in Remote Sensing Scene Classification
This repository is the implementation of our paper: 'D3R-Net: Denoising Diffusion-based Defense Restore Network for Adversarial Defense in Remote Sensing Scene Classification'.

Abstract— In this study, we propose an effective denoising diffusion-based
12 defense restore network (D3R-Net) based on the denoising diffu13 sion model from the perspective of adversarial restoration, which transforms the adversarial examples into clean samples. Utilizing a highly effective denoising diffusion probabilistic model, our D3R-Net transforms input adversarial examples into a state of noise, where diverse forms of adversarial noise transition into Gaussian noise. Subsequently, it captures semantic information through a series of iterative denoising steps. The pixel distribution of adversarial examples is restored in the proposed network to match the original distribution, which enables the classifier to identify adversarial examples correctly. Furthermore, we intro23 duce a combined filtering module to preserve the rich semantic information of the original image, thereby further enhancing the defensive performance of the network. Instead of modifying the model structure or excluding suspected samples, the proposed method restores the adversarial examples, making it simple yet effective and applicable to a broader range of scenarios

![image](figure/framework.png)
## Platform

- Ubuntu 18.04.6 LTS,  torch 2.2.2+cu118, NVIDIA GeForce RTX 3080 Ti 12GB

## Datasets and Weights
You can download the adversarial examples dataset from https://github.com/YonghaoXu/UAE-RS/tree/main.

**Download links:** **[Google Drive](https://drive.google.com/file/d/1tbRSDJwhpk-uMYk2t-RUgC07x2wyUxAL/view?usp=sharing)**    **[Baidu NetDisk](https://pan.baidu.com/s/12SK3jfQ8-p_gU87YVEFMtw)** (Code: 8g1r)

You can download Diffusion model weights.

**Download links:**  **[Baidu NetDisk](https://pan.baidu.com/s/12SK3jfQ8-p_gU87YVEFMtw)** (Code: wp6a)

## Usage

1. Repair adversarial examples, and save the results in the 'result' folder.

 ```py
 python restore.py
 ```

2. Test the OA.

```python 
python test_acc.py
```


## Citation

 Thanks for your attention!
