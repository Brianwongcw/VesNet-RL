# Dreamer from Demonstrations (DreamerfD): Integrating DreamerV2 with Demonstrations

## Overview

We integrate [Dreamerv2](https://github.com/danijar/dreamerv2.git) with demonstrations to maximally improve learning efficiency.
## Method 

Compared with Dreamerv2, we have the following improvements/novelties:

- Demonstration-guided wolrd model learning and policy learning:
    Off-policy demonstration data is sampled by an additional Replay Buffer for world model learning and behavior learning.
<p style="text-align:center;">
    <img src="doc/media/DreamerBC_1.png" width=270 title="xxx" class="center">
    <img src="doc/media/DreamerBC_2.png" width=250>
</p>

- Behaviour cloning loss to regress actions of demonstration trajectories.
  <p style="text-align:center;">
    <img src="doc/media/DreamerBC_3.png" width=270>
</p>

For more information:
- [Project Website](https://sites.google.com/view/dreamerfd/home)


## Install

- Download
  ```sh
  cd ~
  git clone https://github.com/linhongbin/DreamerfD
  ```

- Install by Anaconda virtual environment:
  
    ```sh
    source ~/anaconda3/bin/activate
    conda create -n dreamer_fd python=3.7 -y
    conda activate dreamer_fd
    pip install tensorflow==2.9.0 tensorflow_probability==0.17.0 protobuf==3.20.1
    cd ~/DreamerfD
    pip install -e .

- Install additional GPU support
    ```sh
    conda install cudnn=8.2 cudatoolkit=11.3 -c anaconda -y
    ```
## Training and Evaluation

For needle picking tasks, please refer to [gym_np.md](doc/gym_np.md)


## Citation

```
@misc{lin2023endtoend,
      title={End-to-End Learning of Deep Visuomotor Policy for Needle Picking}, 
      author={Hongbin Lin and Bin Li and Xiangyu Chu and Qi Dou and Yunhui Liu and Kwok Wai Samuel Au},
      year={2023},
      eprint={2303.03675},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```