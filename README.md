# mujoco-2.1-rl-project
Class project for MATH5001 - Mathematics of Machine Learning

Todo:
- [X] Give installation instructions / Describe environment
- [X] Give basic example of rendering environment
- [ ] Decide on deep learning framework / version
- [ ] Decide on [algorithm(s)](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html) and make justification(s) 
- [ ] Find some links / tutorials to guide the project
- [ ] Get *bad* model by Saturday, Nov 20th
- [ ] Get improved model by Tuesday, Nov 23rd
- [ ] Get final model for presentation before Tuesday, Nov 30th

<br>

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

## Install Anaconda (or miniconda)

`conda` is used to manage dependencies in the development environment. Install Anaconda then use the provided environment file.

```sh
conda install python==3.8.12
conda install -c conda-forge gym[all]
pip3 install mujoco-py>=2.1.2.14
```

## Verify MuJoCo installation

```sh
python3 ./mujoco_render/render_test.py
```

If the installation is successful you should see the model make random movements for several seconds