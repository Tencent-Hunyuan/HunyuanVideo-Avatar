<!-- ## **HunyuanVideo-Avatar** -->

<p align="center">
  <img src="assets/material/logo.png"  height=100>
</p>

# **HunyuanVideo-Avatar** 🌅
 
<div align="center">
  <a href="https://github.com/Tencent/HunyuanVideo-Avatar"><img src="https://img.shields.io/static/v1?label=HunyuanVideo-Avatar%20Code&message=Github&color=blue"></a> &ensp;
  <a href="https://HunyuanVideo-Avatar.github.io/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Web&color=green"></a> &ensp;
  <a href="https://hunyuan.tencent.com/modelSquare/home/play?modelId=126"><img src="https://img.shields.io/static/v1?label=Playground&message=Web&color=green"></a>
</div>

<div align="center">
  <a href="https://arxiv.org/pdf/2505.20156"><img src="https://img.shields.io/badge/ArXiv-2505.20156-red"></a> &ensp;
</div>

<div align="center">
  <a href="https://huggingface.co/tencent/HunyuanVideo-Avatar"><img src="https://img.shields.io/static/v1?label=HunyuanVideo-Avatar&message=HunyuanVideo-Avatar&color=yellow"></a> &ensp;
</div>

![image](assets/material/teaser.png)

> [**HunyuanVideo-Avatar: High-Fidelity Audio-Driven Human Animation for Multiple Characters**](https://arxiv.org/pdf/2505.20156) <be>

## 🔥🔥🔥 News!!
* Jun 06, 2025: 🔥 HunyuanVideo-Avatar supports **Single GPU** with only **10GB VRAM**, with **TeaCache** included, **HUGE THANKS** to [Wan2GP](https://github.com/deepbeepmeep/Wan2GP)
* May 28, 2025: 🔥 HunyuanVideo-Avatar is available in Cloud-Native-Build (CNB) [HunyuanVideo-Avatar](https://cnb.cool/tencent/hunyuan/HunyuanVideo-Avatar).
* May  28, 2025: 👋 We release the inference code and model weights of HunyuanVideo-Avatar. [Download](weights/README.md).


## 📑 Open-source Plan

- HunyuanVideo-Avatar
  - [x] Inference 
  - [x] Checkpoints
  - [ ] ComfyUI

## Contents
- [**HunyuanVideo-Avatar** 🌅](#HunyuanVideo-Avatar-)
  - [🔥🔥🔥 News!!](#-news)
  - [📑 Open-source Plan](#-open-source-plan)
  - [Contents](#contents)
  - [**Abstract**](#abstract)
  - [**HunyuanVideo-Avatar Overall Architecture**](#HunyuanVideo-Avatar-overall-architecture)
  - [🎉 **HunyuanVideo-Avatar Key Features**](#-HunyuanVideo-Avatar-key-features)
    - [**Multimodal Video customization**](#multimodal-video-customization)
    - [**Various Applications**](#various-applications)
  - [📈 Comparisons](#-comparisons)
  - [📜 Requirements](#-requirements)
  - [🛠️ Dependencies and Installation](#️-dependencies-and-installation)
    - [Installation Guide for Linux](#installation-guide-for-linux)
  - [🧱 Download Pretrained Models](#-download-pretrained-models)
  - [🚀 Parallel Inference on Multiple GPUs](#-parallel-inference-on-multiple-gpus)
  - [🔑 Single-gpu Inference](#-single-gpu-inference)
    - [Run with very low VRAM](#run-with-very-low-vram)
  - [Run a Gradio Server](#run-a-gradio-server)
  - [🔗 BibTeX](#-bibtex)
  - [Acknowledgements](#acknowledgements)
---

## **Abstract**

Recent years have witnessed significant progress in audio-driven human animation. However, critical challenges remain in (i) generating highly dynamic videos while preserving character consistency, (ii) achieving precise emotion alignment between characters and audio, and (iii) enabling multi-character audio-driven animation. To address these challenges, we propose HunyuanVideo-Avatar, a multimodal diffusion transformer (MM-DiT)-based model capable of simultaneously generating dynamic, emotion-controllable, and multi-character dialogue videos. Concretely, HunyuanVideo-Avatar introduces three key innovations: (i) A character image injection module is designed to replace the conventional addition-based character conditioning scheme, eliminating the inherent condition mismatch between training and inference. This ensures the dynamic motion and strong character consistency; (ii) An Audio Emotion Module (AEM) is introduced to extract and transfer the emotional cues from an emotion reference image to the target generated video, enabling fine-grained and accurate emotion style control; (iii) A Face-Aware Audio Adapter (FAA) is proposed to isolate the audio-driven character with latent-level face mask, enabling independent audio injection via cross-attention for multi-character scenarios. These innovations empower HunyuanVideo-Avatar to surpass state-of-the-art methods on benchmark datasets and a newly proposed wild dataset, generating realistic avatars in dynamic, immersive scenarios. The source code and model weights will be released publicly.

## **HunyuanVideo-Avatar Overall Architecture**

![image](assets/material/method.png)

We propose **HunyuanVideo-Avatar**, a multi-modal diffusion transformer(MM-DiT)-based model capable of generating **dynamic**, **emotion-controllable**, and **multi-character dialogue** videos.

## 🎉 **HunyuanVideo-Avatar Key Features**

![image](assets/material/demo.png)

### **High-Dynamic and Emotion-Controllable Video Generation**

HunyuanVideo-Avatar supports animating any input **avatar images** to **high-dynamic** and **emotion-controllable** videos with simple **audio conditions**. Specifically, it takes as input **multi-style** avatar images at **arbitrary scales and resolutions**. The system supports multi-style avatars encompassing photorealistic, cartoon, 3D-rendered, and anthropomorphic characters. Multi-scale generation spanning portrait, upper-body and full-body. It generates videos with high-dynamic foreground and background, achieving superior realistic and naturalness. In addition, the system supports controlling facial emotions of the characters conditioned on input audio. 

### **Various Applications**

HunyuanVideo-Avatar supports various downstream tasks and applications. For instance, the system generates talking avatar videos, which could be applied to e-commerce, online streaming, social media video production, etc. In addition, its multi-character animation feature enlarges the application such as video content creation, editing, etc. 

## 📜 Requirements

* An NVIDIA GPU with CUDA support is required. 
  * The model is tested on a machine with 8GPUs.
  * **Minimum**: The minimum GPU memory required is 24GB for 704px768px129f but very slow.
  * **Recommended**: We recommend using a GPU with 96GB of memory for better generation quality.
  * **Tips**: If OOM occurs when using GPU with 80GB of memory, try to reduce the image resolution. 
* Tested operating system: Linux


## 🛠️ Dependencies and Installation

Begin by cloning the repository:
```shell
git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar.git
cd HunyuanVideo-Avatar
```

### Installation Guide for Linux

We recommend CUDA versions 12.4 or 11.8 for the manual installation.

Conda's installation instructions are available [here](https://docs.anaconda.com/free/miniconda/index.html).

```shell
# 1. Create conda environment
conda create -n HunyuanVideo-Avatar python==3.10.9

# 2. Activate the environment
conda activate HunyuanVideo-Avatar

# 3. Install PyTorch and other dependencies using conda
# For CUDA 11.8
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda==11.8 -c pytorch -c nvidia
# For CUDA 12.4
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda==12.4 -c pytorch -c nvidia

# 4. Install pip dependencies
python -m pip install -r requirements.txt
# 5. Install flash attention v2 for acceleration (requires CUDA 11.8 or above)
python -m pip install ninja
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3
```

In case of running into float point exception(core dump) on the specific GPU type, you may try the following solutions:

```shell
# Option 1: Making sure you have installed CUDA 12.4, CUBLAS>=12.4.5.8, and CUDNN>=9.00 (or simply using our CUDA 12 docker image).
pip install nvidia-cublas-cu12==12.4.5.8
export LD_LIBRARY_PATH=/opt/conda/lib/python3.8/site-packages/nvidia/cublas/lib/

# Option 2: Forcing to explicitly use the CUDA 11.8 compiled version of Pytorch and all the other packages
pip uninstall -r requirements.txt  # uninstall all packages
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install ninja
pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3
```

Additionally, you can also use HunyuanVideo Docker image. Use the following command to pull and run the docker image.

```shell
# For CUDA 12.4 (updated to avoid float point exception)
docker pull hunyuanvideo/hunyuanvideo:cuda_12
docker run -itd --gpus all --init --net=host --uts=host --ipc=host --name hunyuanvideo --security-opt=seccomp=unconfined --ulimit=stack=67108864 --ulimit=memlock=-1 --privileged hunyuanvideo/hunyuanvideo:cuda_12
pip install gradio==3.39.0 diffusers==0.33.0 transformers==4.41.2

# For CUDA 11.8
docker pull hunyuanvideo/hunyuanvideo:cuda_11
docker run -itd --gpus all --init --net=host --uts=host --ipc=host --name hunyuanvideo --security-opt=seccomp=unconfined --ulimit=stack=67108864 --ulimit=memlock=-1 --privileged hunyuanvideo/hunyuanvideo:cuda_11
pip install gradio==3.39.0 diffusers==0.33.0 transformers==4.41.2
```


## 🧱 Download Pretrained Models

The details of download pretrained models are shown [here](weights/README.md).

## 🚀 Parallel Inference on Multiple GPUs

For example, to generate a video with 8 GPUs, you can use the following command:

```bash
cd HunyuanVideo-Avatar

JOBS_DIR=$(dirname $(dirname "$0"))
export PYTHONPATH=./
export MODEL_BASE="./weights"
checkpoint_path=${MODEL_BASE}/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt

torchrun --nnodes=1 --nproc_per_node=8 --master_port 29605 hymm_sp/sample_batch.py \
    --input 'assets/test.csv' \
    --ckpt ${checkpoint_path} \
    --sample-n-frames 129 \
    --seed 128 \
    --image-size 704 \
    --cfg-scale 7.5 \
    --infer-steps 50 \
    --use-deepcache 1 \
    --flow-shift-eval-video 5.0 \
    --save-path ${OUTPUT_BASEPATH} 
```

## 🔑 Single-gpu Inference

For example, to generate a video with 1 GPU, you can use the following command:

```bash
cd HunyuanVideo-Avatar

JOBS_DIR=$(dirname $(dirname "$0"))
export PYTHONPATH=./

export MODEL_BASE=./weights
OUTPUT_BASEPATH=./results-single
checkpoint_path=${MODEL_BASE}/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt

export DISABLE_SP=1 
CUDA_VISIBLE_DEVICES=0 python3 hymm_sp/sample_gpu_poor.py \
    --input 'assets/test.csv' \
    --ckpt ${checkpoint_path} \
    --sample-n-frames 129 \
    --seed 128 \
    --image-size 704 \
    --cfg-scale 7.5 \
    --infer-steps 50 \
    --use-deepcache 1 \
    --flow-shift-eval-video 5.0 \
    --save-path ${OUTPUT_BASEPATH} \
    --use-fp8 \
    --infer-min
```

### Run with very low VRAM

```bash
cd HunyuanVideo-Avatar

JOBS_DIR=$(dirname $(dirname "$0"))
export PYTHONPATH=./

export MODEL_BASE=./weights
OUTPUT_BASEPATH=./results-poor

checkpoint_path=${MODEL_BASE}/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt

export CPU_OFFLOAD=1
CUDA_VISIBLE_DEVICES=0 python3 hymm_sp/sample_gpu_poor.py \
    --input 'assets/test.csv' \
    --ckpt ${checkpoint_path} \
    --sample-n-frames 129 \
    --seed 128 \
    --image-size 704 \
    --cfg-scale 7.5 \
    --infer-steps 50 \
    --use-deepcache 1 \
    --flow-shift-eval-video 5.0 \
    --save-path ${OUTPUT_BASEPATH} \
    --use-fp8 \
    --cpu-offload \
    --infer-min
```

### Run with 10GB VRAM GPU (TeaCache supported)

Thanks to [Wan2GP](https://github.com/deepbeepmeep/Wan2GP), HunyuanVideo-Avatar now supports single GPU mode with even lower VRAM (10GB) without quality degradation. Check out this [great repo](https://github.com/deepbeepmeep/Wan2GP/tree/main/hyvideo).


## Run a Gradio Server
```bash
cd HunyuanVideo-Avatar

bash ./scripts/run_gradio.sh

```

## 🔗 BibTeX

If you find [HunyuanVideo-Avatar](https://arxiv.org/pdf/2505.20156) useful for your research and applications, please cite using this BibTeX:

```BibTeX
@misc{hu2025HunyuanVideo-Avatar,
      title={HunyuanVideo-Avatar: High-Fidelity Audio-Driven Human Animation for Multiple Characters}, 
      author={Yi Chen and Sen Liang and Zixiang Zhou and Ziyao Huang and Yifeng Ma and Junshu Tang and Qin Lin and Yuan Zhou and Qinglin Lu},
      year={2025},
      eprint={2505.20156},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/pdf/2505.20156}, 
}
```

## Acknowledgements

We would like to thank the contributors to the [HunyuanVideo](https://github.com/Tencent/HunyuanVideo), [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [FLUX](https://github.com/black-forest-labs/flux), [Llama](https://github.com/meta-llama/llama), [LLaVA](https://github.com/haotian-liu/LLaVA), [Xtuner](https://github.com/InternLM/xtuner), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) repositories, for their open research and exploration. 
