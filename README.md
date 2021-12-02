# mujoco-2.1-rl-project
Implementations of different deep reinforcement learning algorithms in PyTorch using OpenAI Gym and MuJoCo as training environments

Algorithms Implemented:
- REINFORCE
- A2C (discrete)
- A2C (continuous)
- DDPG
- SAC

Environments used:
- `CartPole-v0`
- `Pendulum-v1`
- `Ant-v3`
- `Humanoid-v3`

## Examples

### Ant-v3

<p float="left">
  <img src="/assets/ddpg-ant-v3-500k.gif" width="400" height="225" alt="DDPG Ant after 500k environment interactions"/>
  <img src="/assets/experiments_sac-ant-v3-exp2_videos_sac-ant-v3-478749.gif" width="400" height="225" alt="SAC Ant after 480k environment interactions">
</p>

### Humanoid-v3:
<p float="left">
    <img src="/assets/sac-humanoid-240k.gif" width="400" height="225" alt="SAC Humanoid after 240k environment interactions"/>
    <img src="/assets/sac-humanoid-1250k.gif" width="400" height="225" alt="SAC Humanoid after 1250k environment interactions"/>
</p>

# Installation Instructions

These instructions assume you are using Ubuntu 20.04 LTS

## Installing dependencies

```sh
sudo apt install build-essential libx11-dev libglew-dev patchelf
```

## Install MuJoCo
MuJoCo is available for [free](https://mujoco.org/download). Download it from their website or run

```sh
wget -c https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O - | tar -xz
mkdir -p ~/.mujoco
mv ./mujoco210 ~/.mujoco
```

Then, configure the following environment variables
```sh
echo export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$USER/.mujoco/mujoco210/bin:/usr/lib/nvidia >> ~/.bashrc
echo export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so >> ~/.bashrc
source ~/.bashrc
```

## Install Python dependencies

`conda` is used to manage dependencies in the development environment. Create an anaconda environment then use install the following packages:

```sh
conda install python==3.8.12
conda install -c conda-forge gym[all]
conda install -c pytorch pytorch
conda install matplotlib 
pip3 install mujoco-py>=2.1.2.14 more_itertools tqdm
```

## Verify MuJoCo installation

```sh
python3 ./examples/mujoco_render/render_test.py
```

If the installation is successful you should see the model make random movements for several seconds
