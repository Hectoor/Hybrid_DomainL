# Hybrid_DomainL

The implementation of the paper "A Hybrid Domain Learning Framework for Unsupervised Semantic Segmentation". 

![](./figs/overview.png)



## Abstract
Supervised semantic segmentation often fails to generalize well in unseen scenarios due to the domain gap between the source and the target domains. Unsupervised domain adaptation is one possible way to solve this problem. However, the existing methods suffer two limitations. First, the number limitation of samples may lead to decreasing generalization. Second, only the source dataset contains pixel-level annotations, which provide stronger supervision in the source domain and result in overfitting to the source domain. To tackle these issues, we propose a hybrid domain learning (HDL) framework where the hybrid domain acts as the intermediate domain between the source domain and the target domain. Specifically, we first generate the hybrid domain feature (HDF) by a deep feature interpolation method and discuss the characteristics of the hybrid domain feature. Then, we further design a triple domain strategy to align the distribution of the source domain, the hybrid domain, and the target domain. The experiments in the tasks of GTA5 to Cityscapes and SYNTHIA to Cityscapes demonstrate that the proposed HDL framework is robust to domain adaptation and outperforms the state-of-the-art approaches.

## Running 

### Pretrained model 
The model can be found in [here](https://drive.google.com/drive/folders/12Ra5T35A5mU1YFcpBiM2FYlUrd30vQ9H?usp=sharing).

### Training
For training, you can execute by:
```bash
python HDL_trainG2C.py --snapshot-dir 'path to save the model' --restore-from 'path to init model'
```

### Evaluation
For evaluation, you can execute by:
```bash
python HDL_evaluate.py --path_file 'path to the model'
```
