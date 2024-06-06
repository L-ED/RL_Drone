# RL_DRONE
Diploma work based on PyBullet physics simulator. Target of this work is to develop position control system using only following input data: 
- linear and angular velocity in drone local coordinates
- projected gravity [orientation analog]
- position displacement

Also this work contains absolute position controllers from 
https://github.com/utiasDSL/gym-pybullet-drones.git 

<img src="files/imu_flight.gif" alt="formation flight" width="700">

## Installation

```sh
git clone https://github.com/L-ED/RL_Drone.git
cd gym-pybullet-drones/

conda create -n drones python=3.10
conda activate drones

pip3 install --upgrade pip
pip3 install -e . # if needed, `sudo apt install build-essentials` to install `gcc` and build `pybullet`

```

## Use

### RL Position Control from local coordinates

```sh
cd gym_pybullet_drones/examples/
python hover/hover_learn_multienv.py
```

### PID position control example 

```sh
cd gym_pybullet_drones/examples/
python3 pid.py
```

### Stable-baselines3 PPO RL example

```sh
cd gym_pybullet_drones/examples/
python3 learn.py
```