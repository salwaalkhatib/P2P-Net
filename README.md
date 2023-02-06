# P2P-Net
<!-- Official implementation of "[Fine-Grained Object Classification via Self-Supervised Pose Alignment](https://arxiv.org/abs/2203.15987)". Accepted to CVPR2022. -->
<!-- ![image](https://github.com/yangxh11/P2P-Net/blob/main/motivation.jpg) -->

<img src="https://github.com/yangxh11/P2P-Net/blob/main/motivation.jpg" width = "600" height = "450" alt="" align=center />


## To Run

### Create symlink to dataset path
Run ```ln -s /PATH/TO/ALL/DATASETS/ ./data/``` \
This will create a ```data``` folder with symbolic link to the dataset directory that you point to. \
Do the same for outputs in /l/users/SOMEWHERE/ to ./output

Copy weights from <a href="https://mbzuaiac-my.sharepoint.com/:f:/g/personal/jameel_hassan_mbzuai_ac_ae/EsJaTJ_aDf1GuuvUyhjIqCkBr6R1ydjyVaQCBd5KTQHjiQ?e=kxlIyi"> here </a>.

### Running evaluation
Run ``` python train.py --dataset_name DATASET_NAME --resume PATH/TO/SAVED/WEIGHTS --eval``` \

DATSET_NAMEs are: air, air+car, foodx for FGVC aircrafts, Aircrafts+Stanford Cars combined and Food101 respectively. 


# Preparation
## Benchmarks

CUB_200_2011 (CUB) - <http://www.vision.caltech.edu/visipedia/CUB-200-2011.html>

Stanford Cars (CAR) - <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>

FGVC-Aircraft (AIR) - <https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>

Unzip benchmarks to "../Data/" (update the variable "data_config" in train.py if necessary). 



# Training and evaluation

We train the model with 4 V100. The valid batch size is 16\*4=64.
```shell
python train.py
```

# Performance

<img src="https://github.com/yangxh11/P2P-Net/blob/main/performance.jpg" width = "600" height = "400" alt="" align=center />

# Citation

```
@article{p2pnet2022,
      title={Fine-Grained Object Classification via Self-Supervised Pose Alignment}, 
      author={Xuhui Yang, Yaowei Wang, Ke Chen, Yong Xu, Yonghong Tian},
      journal={arXiv preprint arXiv:2203.15987},
      year={2022},
}
```

# Acknowledgement

This work is supported by the China Postdoctoral Science Foundation (2021M691682), the National Natural Science Foundation of China (61902131, 62072188, U20B2052), the Program for Guangdong Introducing Innovative and Entrepreneurial Teams (2017ZT07X183), and the Project of Peng Cheng Laboratory (PCL2021A07).
