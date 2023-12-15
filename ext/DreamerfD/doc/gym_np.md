## Install gym_np benchmark

  - make sure install [DreamerfD](https://github.com/linhongbin/DreamerfD)

  - Download and install [gym_np](https://github.com/linhongbin/gym-np.git):
  
    ```sh
    # Download
    conda activate dreamer_fd
    cd ~
    git clone https://github.com/linhongbin/gym-np.git
    cd gym-np
    git submodule init
    git submodule update

    # update setting.sh
    printf "ENV_NAME=\"dreamer_fd\"\n\
    ANACONDA_PATH=\"${HOME}/anaconda3\"\n\
    AMBF_SRC_PATH=\"${HOME}/gym-np/ext/ambf\"\n\
    AMBF_BUILD_PATH=\"${HOME}/gym-np/build/ambf\"\n"\
    > ~/gym-np/bash/setting.sh

    # install
    source ~/anaconda3/bin/activate
    conda activate dreamer_fd
    source ~/gym-np/bash/setting.sh # claim setting variables
    source ~/gym-np/bash/install/ambf.sh # install AMBF, will take some time to compile
    source ~/gym-np/bash/install/other.sh # install other packages
    ```
  

## Launch simulator

- **Launch roscore, open 1st terminal and type**
  ```sh
  roscore
  ```

- **Launch simulator, open 2st terminal and type**
  ```sh
  cd ~/gym-np
  source ./bash/ambf/sim_standard.sh # For standard needle
  ```
  > For needle variations, type the following commands instead:
  >
  >  ```sh
  >  source ./bash/ambf/sim_small.sh # >For small needle
  >  ```
  >  ```sh
  >  source ./gym-np/bash/ambf/sim_large.sh # For large needle
  >  ```
  >  ```sh
  >  source ./gym-np/bash/ambf/sim_irregular1.sh # For irregular shape 1
  >  ```
  >  ```sh
  >  source ./gym-np/bash/ambf/sim_irregular2.sh # For irregular shape 2
  >  ```

- **Launch crtk interface for control, open 3st terminal and type**
  ```sh
  cd ~/gym-np
  source ./bash/ambf/crtk.sh
  ```

- **Reset robotic arm to ready position, open 4st terminal and type**
  ```sh
  cd ~/gym-np
  source ./bash/ambf/reset.sh 
  ```
## Training

- Initialize DreamerfD environment for gym_np
  ```sh
  cd ~/gym-np
  source ~/gym-np/bash/ambf/init.sh # init conda env and other environment variables
  cd ~/DreamerfD
  ```

- Training baselines:
  - **Train with DreamerfD**
  ```sh
  python ./example/suture_json.py --section 1 --seed 1
  ```
  It will take 3 days on RTX3070 GPU for convergence of evaluated succeses rate (reaching 80%).
  > Resuming training is supported. To resume training, you need to use different seed number and use the same section number w.r.t. your preivous training, take the above command for example, you can run
  > ```sh
  > python ./example/suture_json.py --section 1 --seed 2
  >```


  - **Train with [Dreamer](https://danijar.com/project/dreamerv2/)**
  ```sh
  python ./example/suture_json.py --json ./example/jsons/suture_np/Dreamer.yaml --section 1 --seed 1
  ```

## Evaluation

  To evaluate a train model, you need to use ```--only-eval``` flag and ```--json``` for the directory to ```<train-log-path>/config.yaml```, for example:

  ```sh
  cd ~/gym-np
  source ~/gym-np/bash/ambf/init.sh # init conda env and other environment variables
  cd ~/DreamerfD
  python ./example/suture_json.py --json ./data/rl_log/ambf-phantom-psm2-segment_script-zoom_needle_boximage-prefill8000-clutch6/DreamerfD/1/config.yaml --only-eval
  ```

## Monitor Training with Tensorboard.

  To monitor the training process and results with tensorboard, type 

  ```sh
  source ~/anaconda3/bin/activate
  conda activate dreamer_fd
  tensorboard --logdir ./log
  ```


## Plotting

  To plot the figures in the paper, you run the scripts in [./example/plot](./example/plot)
  ```sh
  source ~/anaconda3/bin/activate
  conda activate dreamer_fd
  python example/plot/fig1.py
  ```


