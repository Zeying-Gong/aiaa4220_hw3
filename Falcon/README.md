<div align="center">
  <h1>From Cognition to Precognition: <br>A Future-Aware Framework for Social Navigation</h1>
  <h3>
    <a href="https://zeying-gong.github.io/">Zeying Gong</a>,
    <a href="https://hutslib.github.io/">Tianshuai Hu</a>,
    <a href="https://openreview.net/profile?id=~Ronghe_Qiu2">Ronghe Qiu</a>,
    <a href="https://junweiliang.me/">Junwei Liang</a>
  </h3>

  <p>
    <a href="https://zeying-gong.github.io/projects/falcon/">Project Website</a> |
    <a href="https://arxiv.org/abs/2409.13244">Paper (ArXiv)</a>
  </p>

  <!-- Badges -->
  <p>
    <a href="https://zeying-gong.github.io/projects/falcon/">
      <img src="https://img.shields.io/badge/Web-Falcon-deepgreen.svg" alt="Falcon Project Web Badge">
    </a>
    <a href="https://www.youtube.com/watch?v=elNI7XlRyvU">
      <img src="https://img.shields.io/badge/Video-Youtube-red.svg" alt="YouTube Video Badge">
    </a>
    <a href="https://arxiv.org/abs/2409.13244">
      <img src="https://img.shields.io/badge/cs.ai-arxiv:2409.13244-42ba94.svg" alt="arXiv Paper Badge">
    </a>
    <a href="https://github.com/facebookresearch/habitat-sim">
      <img src="https://img.shields.io/static/v1?label=supports&message=Habitat%20Sim&color=informational" alt="Habitat Sim Badge">
    </a>
    <a href="https://github.com/Zeying-Gong/habitat-lab/blob/main/LICENSE">
      <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License Badge">
    </a>
  </p>

  <!-- Task Illustration Image -->
  <img src="web-img/task_illustration_v5.png" width="600" alt="Task Illustration">
</div>


## :sparkles: Overview

To navigate safely and efficiently in crowded spaces, robots should not only perceive the current state of the environment but also anticipate future human movements.
In this paper, we propose a reinforcement learning architecture, namely **Falcon**, to tackle socially-aware navigation by explicitly predicting human trajectories and penalizing actions that block future human paths.
To facilitate realistic evaluation, we introduce a novel SocialNav benchmark containing two new datasets, [**Social-HM3D & Social-MP3D**](#3-downloading-the-social-hm3d--social-mp3d-datasets).
This benchmark offers large-scale photo-realistic indoor scenes populated with a reasonable amount of human agents based on scene area size, incorporating natural human movements and trajectory patterns.
We conduct a detailed experimental analysis with the state-of-the-art [**learning-based method**](#two-classic-rule-based-methods-astar--orca) and two classic [**rule-based path-planning algorithms**](#two-rl-based-methods-proximity--falconours) on the new benchmark.
The results demonstrate the importance of future prediction and our method achieves the best task success rate of 55% while maintaining about 90% personal space compliance.

## :hammer_and_wrench: Installation

### Getting Started

#### 1. **Preparing conda env**

Assuming you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) installed, let's prepare a conda env:
```
conda_env_name=falcon
conda create -n $conda_env_name python=3.9 cmake=3.14.0
conda activate $conda_env_name
```

#### 2. **conda install habitat-sim & habitat-lab**
Following [Habitat-lab](https://github.com/facebookresearch/habitat-lab.git)'s instruction:
```
conda install habitat-sim=0.3.1 withbullet headless -c conda-forge -c aihabitat
```

If you encounter network problems, you can manually download the Conda package from [this link](https://anaconda.org/aihabitat/habitat-sim/0.3.1/download/linux-64/habitat-sim-0.3.1-py3.9_headless_bullet_linux_3d6d67d6deae4ab2472cc84df7a3cef1503f606d.tar.bz2) to download the conda bag, and install it via: `conda install --use-local /path/to/xxx.tar.bz2` to download.

Then, assuming you have this repositories cloned (forked from Habitat 3.0), install necessary dependencies of Habitat.
```
git clone https://github.com/Zeying-Gong/Falcon.git
cd Falcon
pip install -e habitat-lab
pip install -e habitat-baselines
pip install -r requirements.txt # install other dependencies
```

#### 3. **Downloading the Social-HM3D & Social-MP3D datasets**

- Download Scene Datasets

Following the instructions for **HM3D** and **MatterPort3D** in Habitat-lab's [Datasets.md](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md).

- Download Episode Datasets

Download social navigation (SocialNav) episodes for the test scenes, which can be found here: [Link](https://drive.google.com/drive/folders/1V0a8PYeMZimFcHgoJGMMTkvscLhZeKzD?usp=drive_link).

After downloading, unzip and place the datasets in the default location:
```
unzip -d data/datasets/pointnav
```
- Download Leg animation

```
wget https://github.com/facebookresearch/habitat-lab/files/12502177/spot_walking_trajectory.csv -O data/robots/spot_data/spot_walking_trajectory.csv
```

- Download Multi-agent necessary data

```
python -m habitat_sim.utils.datasets_download --uids hab3-episodes habitat_humanoids hab3_bench_assets hab_spot_arm
```

The file structure should look like this:

```
Falcon/
└── data/
    ├── datasets
    │   └── pointnav
    │       ├── social-hm3d
    │       │   ├── train
    │       │   │   ├── content
    │       │   │   └── train.json.gz
    │       │   └── val
    │       │       ├── content
    │       │       └── val.json.gz
    │       └── social-mp3d
    │           ├── train
    │           │   ├── content
    │           │   └── train.json.gz
    │           └── val
    │               ├── content
    │               └── val.json.gz
    ├── scene_datasets
    ├── robots
    ├── humanoids
    ├── versoned_data
    └── hab3_bench_assets
```

Note that here the definition of SocialNav is different from the original task in [Habitat 3.0](https://arxiv.org/abs/2310.13724).



## :arrow_forward: Evaluation

### Two classic rule-based methods (ASTAR & ORCA)

In this paper, two rule-based methods are used for evaluation:

- **[ASTAR](https://ieeexplore.ieee.org/document/4082128)**: A well-known pathfinding algorithm that finds the shortest path using a heuristic to estimate the cost.

- **[ORCA](https://gamma.cs.unc.edu/ORCA/publications/ORCA.pdf)**: A multi-agent navigation algorithm designed for collision-free movement through reciprocal avoidance.

You can evaluate Astar or ORCA on the Social-HM3D or Social-MP3D datasets using the following template:

```
python -u -m habitat-baselines.habitat_baselines.run \
--config-name=social_nav_v2/<algorithm>_<dataset>.yaml
```
For example, to run Astar on the Social-HM3D dataset:

```
python -u -m habitat-baselines.habitat_baselines.run \
--config-name=social_nav_v2/astar_hm3d.yaml
```

If you wish to generate videos, simply add the `habitat_baselines.eval.video_option=["disk"]` to the end of the command. For instance, to run Astar on the Social-HM3D dataset and record videos:

```
python -u -m habitat-baselines.habitat_baselines.run \
--config-name=social_nav_v2/astar_hm3d.yaml \
habitat_baselines.eval.video_option=["disk"]
```

### RL-based methods Proximity

The code of Proximity can be found in this [link](https://github.com/EnricoCancelli/ProximitySocialNav).

### Falcon (ours)

The pretrained models can be found in [this link](https://drive.google.com/drive/folders/1Bx1L9U345P_9pUfADk3Tnj7uK01EpxZY?usp=sharing). Download it to the root directory.

You can evaluate it on the Social-HM3D or Social-MP3D datasets using the following template:

```
python -u -m habitat-baselines.habitat_baselines.run \
--config-name=social_nav_v2/falcon_<dataset>.yaml
```

For example, to run it on the Social-HM3D dataset:

```
python -u -m habitat-baselines.habitat_baselines.run \
--config-name=social_nav_v2/falcon_hm3d.yaml
```

## :rocket: Training

To reproduce our training, use the following command for single-gpu setup:

```
python -u -m habitat-baselines.habitat_baselines.run \
--config-name=social_nav_v2/falcon_hm3d_train.yaml
```

for multi-gpu training, use:

```
sh habitat-baselines/habitat_baselines/rl/ddppo/single_node_falcon.sh
```

Note: The training was performed using **4x NVIDIA RTX 3090 GPUs**, and it took approximately **2 days**. 

## :black_nib: Citation

If you find this repository useful in your research, please consider citing our paper:

```
@article{gong2024cognition,
  title={From Cognition to Precognition: A Future-Aware Framework for Social Navigation},
  author={Gong, Zeying and Hu, Tianshuai and Qiu, Ronghe and Liang, Junwei},
  journal={arXiv preprint arXiv:2409.13244},
  year={2024}
}
```

## 	:pray: Acknowledgments

We would like to thank the following repositories for their contributions:
- [Proximity](https://github.com/EnricoCancelli/ProximitySocialNav)
- [VLFM](https://github.com/bdaiinstitute/vlfm/tree/main)
- [Habitat-Lab](https://github.com/facebookresearch/habitat-lab)
- [Habitat-Sim](https://github.com/facebookresearch/habitat-sim)