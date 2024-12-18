# Self-Debiasing Calibration (SDC)
Data and code for paper titled [Unleashing the Potential of Model Bias for Generalized Category Discovery](https://arxiv.org/abs/2412.12501) (AAAI 2025 paper)

*Generalized Category Discovery (GCD)* is a significant and complex task that aims to identify both known and undefined novel categories from a set of unlabeled data, leveraging another labeled dataset containing only known categories. The primary challenges stem from model bias induced by pre-training on only known categories and the lack of precise supervision for novel ones, leading to *category bias* towards known categories and *category confusion* among different novel categories. To address these challenges, we propose a novel framework named *Self-Debiasing Calibration (SDC)*, which provides a novel insight into unleashing the potential of the bias to facilitate novel category learning. SDC dynamically adjusts the output logits of the current training model using the output of the biased model. This approach produces less biased logits to effectively address the issue of category bias towards known categories, and generates more accurate pseudo labels for unlabeled data, thereby mitigating category confusion for novel categories. Experiments on three benchmark datasets show that SDC outperforms SOTA methods, especially in the identification of novel categories.


## Contents
[1. Data](#data)

[2. Model](#model)

[3. Requirements](#requirements)

[4. Running](#running)

[5. Results](#results)

[6. Thanks](#thanks)

[7. Citation](#citation)

## Data
We performed experiments on three public datasets: [clinc](https://aclanthology.org/D19-1131/), [banking](https://aclanthology.org/2020.nlp4convai-1.5/) and [hwu](https://arxiv.org/abs/1903.05566), which have been included in our repository in the data folder ' ./data '.

## Model
An overview of our model is shown in the figure.
<div align=center>
<img src="./figures/model.png"/>
</div>

## Requirements
* python==3.8
* pytorch==1.12.0
* transformers==4.26.1
* scipy==1.10.1
* numpy==1.23.5
* scikit-learn==1.2.1

## Running
Pre-training, training and testing our model through the bash scripts:
```
sh run.sh
```
You can also add or change parameters in run.sh (More parameters are listed in init_parameter.py)

## Results
<div align=center>
<img src="./figures/results.png"/>
</div>
It should be noted that the experimental results may be different because of the randomness of clustering when testing even though we fixed the random seeds.

## Thanks
Some code references the following repositories:
* [KTN](https://github.com/yibai-shi/KTN)
* [DPN](https://github.com/Lackel/DPN)
* [TAN](https://github.com/Lackel/TAN)


## Citation
If our paper or code is helpful to you, please consider citing our paper:
```
@misc{an2024unleashingpotentialmodelbias,
      title={Unleashing the Potential of Model Bias for Generalized Category Discovery}, 
      author={Wenbin An and Haonan Lin and Jiahao Nie and Feng Tian and Wenkai Shi and Yaqiang Wu and Qianying Wang and Ping Chen},
      year={2024},
      eprint={2412.12501},
      archivePrefix={arXiv},
      primaryClass={cs.LG}, 
}
```
