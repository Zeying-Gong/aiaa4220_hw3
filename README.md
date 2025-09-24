# AIAA 4220 - Homework 3: Social Navigation with 3D Simulator

[![Web](https://img.shields.io/badge/Web-Falcon-blue)](https://zeying-gong.github.io/projects/falcon/)
[![Paper](https://img.shields.io/badge/ICRA-2025-red)](https://ieeexplore.ieee.org/abstract/document/11127910/)
[![EvalAI](https://img.shields.io/badge/Submit-EvalAI-green)](https://eval.ai/web/challenges/challenge-page/2650/overview)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](LICENSE)

This is the homework 3 for AIAA 4220. We will perform a **social navigation task** on the Habitat simulator.

The assignment challenges students to develop advanced **RGBD-based perception and navigation systems** for autonomous agents operating in **dynamic, human-populated indoor environments**. The goal is to create agents that **navigate efficiently** while adhering to **social norms**—avoiding collisions and respecting personal space.

## 📋 Table of Contents

- [🔗 Prerequisites](#-prerequisites)
- [📚 Dataset](#-dataset)
- [📥 Setup](#-setup)
- [📊 Evaluation Metrics](#-evaluation-metrics)
- [🚀 Submission Guidelines](#-submission-guidelines)
- [🔍 Q&A](#-qa)
- [👏 Citation](#-citation)

## 🔗 Prerequisites

Our Docker image uses the headless version. Habitat-sim version should be at least **Habitat 3.0** for multi-agent support.

### System Requirements
* **OS:** Ubuntu 18.04+, macOS
* **Python:** Python >= 3.9
* **Build Tools:** CMake >= 3.10
* **System RAM:** 50GB+ recommended (for storing datasets and large scenes)
* **GPU Memory:** 
  - **Mini Models:** 8GB+ VRAM minimum
  - **Full Models:** 24GB+ VRAM recommended
* **Disk Space:** ~30GB for full dataset and dependencies

## 📚 Dataset

This homework uses the subset of the **Social-HM3D** benchmark and provides:

- **Goal-driven Trajectories**: Humans navigate with intent, avoiding random or repetitive paths
- **Natural Behaviors**: Movement includes walking, pausing, and realistic avoidance via ORCA
- **Balanced Density**: Human count is scaled to scene size, avoiding over- or under-crowding
- **Diverse Environments**: Includes 844 scenes for Social-HM3D

### Dataset Statistics

| Dataset         | Num. of Scenes | Scene Types                   | Num. of Humans |
|----------------|----------------|-------------------------------|------------|
| **Social-HM3D** | 844            | Residence, Office, Shop, etc. | 0–6        | 

### Dataset Preparation

**⚠️ Important:** You must obtain API credentials for dataset download. Contact the instructors if you need assistance.

1. **Download Scene Datasets**

   Follow the instructions for **HM3D** in Habitat-lab's [Datasets.md](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md). By default, the download script fetches version 0.2 of the HM3D dataset.

   ```bash
   python -m habitat_sim.utils.datasets_download --username <api-token-id> --password <api-token-secret> --uids hm3d
   ```

2. **Download Task Datasets**

   Download social navigation episodes for evaluations in [HuggingFace Page](https://huggingface.co/datasets/zgong313/aiaa4220-social-nav-dataset)

   ```bash
   # After downloading, unzip and place in the default location
   unzip -d data/datasets/pointnav
   ```

3. **Download Leg Animation Data**

   ```bash
   wget https://github.com/facebookresearch/habitat-lab/files/12502177/spot_walking_trajectory.csv -O data/robots/spot_data/spot_walking_trajectory.csv
   ```

4. **Download Multi-agent Dependencies**

   ```bash
   python -m habitat_sim.utils.datasets_download --uids habitat_humanoids hab3_bench_assets hab_spot_arm
   ```

### Expected File Structure

```
aiaa4220_hw3/Falcon/
└── data/
    ├── datasets/
    │   └── pointnav/
    │       └── social-hm3d/
    │           ├── train/
    │           │   ├── content/
    │           │   └── train.json.gz
    │           └── val/
    │               ├── content/
    │               └── val.json.gz
    ├── scene_datasets/
    ├── robots/
    ├── humanoids/
    ├── versioned_data/
    └── hab3_bench_assets/
```

**Note:** The SocialNav definition here differs from the original task in [Habitat 3.0](https://arxiv.org/abs/2310.13724).

## 📥 Setup

This baseline is based on **Falcon** - *"From Cognition to Precognition: A Future-Aware Framework for Social Navigation"* ([GitHub Repository](https://github.com/Zeying-Gong/Falcon)).

### Environment Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Zeying-Gong/aiaa4220_hw3.git --recursive
   cd aiaa4220_hw3/Falcon
   ```

2. **Docker Setup (Recommended)**
   
   We provide a standardized Docker environment to ensure consistency across submissions.

   ```bash
   # Login to DockerHub
   docker login
   docker pull zeyinggong/robosense_socialnav:v0.7
   
   # Alternative for network issues:
   # docker login quay.io -u <your-quay-username>
   # docker pull quay.io/zeyinggong/robosense_socialnav:v0.7
   ```

3. **Run the Container**
   ```bash
   docker run --name homework -it --gpus all --network host \
     --runtime=nvidia \
     --entrypoint /bin/bash \
     -w /app/Falcon \
     -v /path/to/Falcon:/app/Falcon \
     zeyinggong/robosense_socialnav:v0.7
   ```

**📝 Note:** Students are strongly encouraged to use this image for local development and testing to ensure compatibility with the remote evaluation environment.

### Training Commands

The baseline offers two model sizes:
- **Full version**: ResNet50 + LSTM512 (requires 24GB+ VRAM)
- **Mini version**: ResNet18 + LSTM128 (requires 8GB+ VRAM)

#### Single-GPU Training

**Full Model:**
```bash
python -u -m habitat_baselines.habitat_baselines.run \
--config-name=social_nav_v2/falcon_hm3d_train.yaml
```

**Mini Model:**
```bash
python -u -m habitat_baselines.habitat_baselines.run \
--config-name=social_nav_v2/falcon_hm3d_train_mini.yaml
```

**For limited VRAM (reduces environments from 8 to 1):**
```bash
python -u -m habitat_baselines.habitat_baselines.run \
--config-name=social_nav_v2/falcon_hm3d_train_mini.yaml \
habitat_baselines.num_environments=1
```

#### Multi-GPU Training

**Full Model:**
```bash
sh habitat-baselines/habitat_baselines/rl/ddppo/single_node_falcon.sh
```

**Mini Model:**
```bash
sh habitat-baselines/habitat_baselines/rl/ddppo/single_node_falcon_mini.sh
```

You can also finetune from the pretrain weights, which can be found in this [link](https://drive.google.com/drive/folders/1Bx1L9U345P_9pUfADk3Tnj7uK01EpxZY?usp=sharing). Download it to the root directory. Remember to set `habitat_baselines.rl.ddppo.pretrained = True`

### Testing Commands

Testing only supports single-GPU evaluation.

**Full Model:**
```bash
python -u -m habitat_baselines.habitat_baselines.run \
--config-name=social_nav_v2/falcon_hm3d.yaml
```

**Mini Model:**
```bash
python -u -m habitat_baselines.habitat_baselines.run \
--config-name=social_nav_v2/falcon_hm3d_mini.yaml
```

**For limited VRAM (reduces environments from 8 to 1):**
```bash
python -u -m habitat_baselines.habitat_baselines.run \
--config-name=social_nav_v2/falcon_hm3d_train_mini.yaml \
habitat_baselines.num_environments=1
```

## 📊 Evaluation Metrics

Our benchmark focuses on two key aspects: **task completion** and **social compliance**.

Your solution will be evaluated using the following metrics:

### ✅ Success Rate (SR)
The percentage of episodes where the robot successfully reaches the goal. Success requires the robot to stop within 1.0 meter of the goal and for the goal to be oracle-visible (i.e., visible from the final pose without further movement).

### ✅ SPL (Success weighted by Path Length)
This metric evaluates path efficiency relative to the shortest possible path. It is defined as:

$$SPL = \frac{1}{N} \sum_{i=1}^{N} S_i \cdot \frac{l_i}{\max(p_i, l_i)}$$

Where:
- `l_i`: Length of the shortest path to the goal in episode *i*
- `p_i`: Path length taken by the agent in episode *i*
- `S_i`: 1 if successful, 0 otherwise

### ✅ PSC (Personal Space Compliance)
PSC is computed as the percentage of timesteps where the robot remains at least **0.5 meters** away from all humans. It reflects socially appropriate behavior. A good policy balances fast goal-reaching with minimizing personal space violations.

### ✅ H-Coll (Human Collision Rate)
Percentage of episodes involving any human collision. Any such collision results in failure and affects SR and PSC.

### 🎯 Total Score
We compute a final weighted score to summarize performance:

$$Total = 0.4 \cdot SR + 0.3 \cdot SPL + 0.3 \cdot PSC$$

This score encourages both efficiency and social awareness.

## 🚀 Submission Guidelines

This homework follows the submission guidelines from [**IROS RoboSense Challenge -- Track 2 Social Navigation**](https://robosense2025.github.io/track2). All submissions are evaluated through the EvalAI platform. Submit your solution as follows: 

### [🎖️ **EvalAI Challenge Entry**](https://eval.ai/web/challenges/challenge-page/2650/overview)


### Submission Format

Submit a single `submission.zip` archive with the following structure:

```
submission.zip
├── run.sh                   # Entry point script (must be executable)
├── config.yaml              # YAML config for inference (optional)
├── model.pth                # Pretrained weights (optional)
├── custom_policy.py         # Your custom policy/inference logic (optional)
├── report.pdf               # Technical report (required)
└── README.md                # Setup and usage instructions
```

**📥 Download submission examples:**
- [Full model submission template](https://drive.google.com/file/d/1k5tMeocASZhCUL2SFUILqWTszNRYjC4L/view?usp=sharing)
- [Mini model submission template](https://drive.google.com/file/d/1IRg5iPrWOOKKL6hTCWzZ2deQPLerBAMX/view?usp=sharing)

### Submission Process

**Phase 1: Minival (Development)**
- Use for quick validation and debugging
- Max submissions/day: 5
- Max total submissions: 60
- Results available within 5-10 minutes

**Phase 2: Test (Final)**
- Final evaluation dataset
- Max submissions/day: 2
- Max total submissions: 20
- Results available within 1-2 hours

### Upload Methods

- **Web UI**: For files ≤ 400 MB via EvalAI platform
- **CLI**: For files > 400 MB using EvalAI command-line tools

### Baseline Performance

The Falcon baseline achieves the following performance on the evaluation set of the homework (100 test episodes):

| Model       | Split       | Success ↑ | SPL ↑  | PSC ↑  | H-Coll ↓ | Total ↑  |
|---------------|-------------|-----------|--------|--------|-----------|--------|
| Mini   | Minival     | 10.00     | 6.72  | 94.87  | 70.00     | 34.48 |
| Full   | Minival     | 50.00     | 48.48  | 89.36  | 40.00     | 61.35 |
| Mini   | Test        | 37.00     | 36.99  | 93.02  | 42.00     | 53.80 |
| Full   | Test        | 53.00     | 48.80  | 91.20  | 41.00     | 63.20 |

## 🔍 Q&A

### 1. Why are my results not showing on the leaderboard?

The most common reason is that your submission did not generate a valid `result.json` file in the expected format. Please ensure your submission:

- Produces a valid JSON file at `output/result.json`
- Contains all required metrics: `SR`, `SPL`, `PSC`, `H-Coll`
- Uses the correct numerical format (floating-point values)

**Example of correct format:**
```json
{
    "SR": 0.50,
    "SPL": 0.45,
    "PSC": 0.60,
    "H-Coll": 0.02
}
```

### 2. How do I handle different dependencies or environment requirements?

If your submission requires additional dependencies, add installation commands to your `run.sh` script using conda/pip.

**Example `run.sh` with dependency installation:**
```bash
#!/bin/bash
# Install additional dependencies
conda install -c conda-forge your_package
pip install specific_library==1.2.3

# Run your inference
source activate falcon
python your_inference_script.py
```

### 3. What is the correct dataset path for my code?

To simplify evaluation, we use a fixed dataset mount point inside the evaluation Docker container:

```
/app/Falcon/data/datasets/pointnav/social-hm3d/minival
```

**Important**: Your code should always read data from this path, regardless of which phase is being evaluated. Our backend automatically mounts the appropriate dataset for each phase to this location.

### 4. What is the action space?

The action set has been refined to clearly distinguish between **stopping** *(ending an episode)* and **pausing** *(remaining stationary)*. A new **move_backward** action is also introduced.

```
0 - stop            # now correctly ends the episode
1 - move_forward
2 - turn_left
3 - turn_right
4 - move_backward   # newly added action for moving backward
5 - pause           # newly added action for pausing without movement
```

The two new actions (4-move_backward, 5-pause) are **optional**. Students can also continue using only actions 0-3 from **pretrained models**.

### 5. How can I test my submission locally?

**For Phase II**, we recommend using Docker image v0.7 for local testing:

```bash
docker run --rm -it \
    --gpus all \
    --runtime=nvidia \
    -w /app/Falcon \
    -v /path/to/your/submission:/app/Falcon/input:ro \
    -v /path/to/your/data:/app/Falcon/data:ro \
    zeyinggong/robosense_socialnav:v0.7
```

You can manually execute your `run.sh` inside the container to verify correctness.

### 6. How long does evaluation take?

- **Minival Phase**: 5–10 minutes
- **Test Phase**: 1-2 hours (depending on number of environments used, queue length, and inference runtime)

### 7. Can I modify the evaluation pipeline or import custom code?

**Yes, we encourage flexible approaches!** You have significant freedom to modify the evaluation pipeline and import your own code. You can base your modifications on:

- `Falcon/habitat-baselines/habitat_baselines/eval.py` (main evaluation script)
- `Falcon/habitat-baselines/habitat_baselines/rl/ppo/falcon_evaluator.py` (evaluator implementation)

You can import and integrate your custom modules, modify the inference pipeline, or adapt the evaluation logic to suit your approach.

**⚠️ However, the following restrictions must be respected:**
1. **No bypassing simulation**: You cannot circumvent the simulator's navigation logic to directly generate result files
2. **No additional sensors/information**: You cannot access extra  information or sensors beyond allowed observations during evaluations
3. **Open-source models only**: You cannot use proprietary pretrained models or private datasets that are not publicly accessible

These restrictions ensure fair competition while maintaining the scientific integrity of the homework.

## 👏 Citation

If you find this repository useful in your research, please consider citing:

```
@inproceedings{gong2025cognition,
  title={From cognition to precognition: A future-aware framework for social navigation},
  author={Gong, Zeying and Hu, Tianshuai and Qiu, Ronghe and Liang, Junwei},
  booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={9122--9129},
  year={2025},
  organization={IEEE}
}

@article{robosense2025track2,
  title     = {RoboSense Challenge 2025: Track 2 - Social Navigation},
  author    = {RoboSense Challenge Organizers},
  booktitle = {IROS 2025},
  year      = {2025},
  url       = {https://robosense2025.github.io/track2}
}
```

