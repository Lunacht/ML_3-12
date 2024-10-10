# CPR & CPR_Improved

原始代码项目为论文《Cross Pairwise Ranking for Unbiased Item Recommendation》的实现，链接为[CPR](https://github.com/Qcactus/CPR)，本项目是在原始代码的基础上做了一些改进（具体见pdf），使得模型具有更高的无偏性。

<!-- ## Citation

If you want to use our codes and datasets in your research, please cite:

```
@inproceedings{cpr22,
  title={Cross Pairwise Ranking for Unbiased Item Recommendation},
  author={Wan, Qi and He, Xiangnan and Wang, Xiang and Wu, Jiancan and Guo, Wei and Tang, Ruiming},
  booktitle={Proceedings of the ACM Web Conference 2022},
  pages={2370--2378},
  year={2022}
}
``` -->


## 环境配置

- python=3.6
- tensorflow-gpu=1.15.5
- numpy
- scipy
- pandas
- Cython

确保在您的环境中已安装 GCC ( 建议使用Linux系统 )。

## 数据集

预处理后的数据已经放置在`data/`文件夹中，包括四个数据集： MovieLens 20M 、Amazon-Book 、Netflix Prize 、Alibaba iFashion 。如果您想了解它们是如何从原始数据生成的，请参阅 [data_preprocess.py](data_preprocess.py) 。

## 编译

代码中的采样器和评估器主要使用 Cython 和 C++ 实现为扩展模块，这比 Python 实现要快得多。运行以下命令以编译所有扩展模块：

```shell
python setup.py build_ext --inplace
```

您可以放心忽略此命令输出中的警告。

## 结果复现

以下是一些用于复现我们的实验报告中结果的命令。

### CPR

以下命令可以复现CPR在4个骨干网络（MF, LightGCN, NeuMF, NGCF）和2个数据集（MovieLens, AmazonBook）上的结果：

#### MF

（MF 相当于 0 层的 LightGCN）

```shell
python CPR.py --dataset movielens_20m --lr 0.0001 --reg 0.001 --ks 20 --batch_size 2048 --n_layer 0 --embed_type lightgcn --sample_rate 3 --sample_ratio 3 --eval_types valid test --eval_epoch 4 --early_stop 10 
```

```shell
python CPR.py --dataset amazonbook --lr 0.0001 --reg 0.001 --ks 20 --batch_size 2048 --n_layer 0 --embed_type lightgcn --sample_rate 3 --sample_ratio 2 --eval_types valid test --eval_epoch 4 --early_stop 10 
```

#### LightGCN

```shell
python CPR.py --dataset movielens_20m --lr 0.001 --reg 0.001 --ks 20 --batch_size 2048 --n_layer 3 --embed_type lightgcn --sample_rate 3 --sample_ratio 3 --eval_types valid test --eval_epoch 1 --early_stop 10 
```

```shell
python CPR.py --dataset amazonbook --lr 0.001 --reg 0.001 --ks 20 --batch_size 2048 --n_layer 3 --embed_type lightgcn --sample_rate 3 --sample_ratio 4=3 --eval_types valid test --eval_epoch 1 --early_stop 10 
```

#### NeuMF

```shell
python CPR.py --dataset movielens_20m --lr 0.0001 --reg 0.001 --weight_reg 0.01 --weight_sizes 256 --ks 20 --batch_size 2048  --n_layer 0 --embed_type lightgcn --sample_rate 3 --sample_ratio 3 --eval_types valid test --eval_epoch 1 --early_stop 10 
```

```shell
python CPR.py --dataset amazonbook --lr 0.001 --reg 0.001 --weight_reg 0.01 --weight_sizes 256 --ks 20 --batch_size 2048 --n_layer 0 --embed_type lightgcn   --sample_rate 3 --sample_ratio 3 --eval_types valid test --eval_epoch 1 --early_stop 10 
```

#### NGCF

```shell
python CPR.py --dataset movielens_20m --lr 0.0001 --reg 0.001 --ks 20 --batch_size 2048 --n_layer 0 --embed_type ngcf --sample_rate 3 --sample_ratio 3 --eval_types valid test --eval_epoch 1 --early_stop 10 
```

```shell
python CPR.py --dataset amazonbook --lr 0.0001 --reg 0.001 --weight_reg 0.001 --ks 20 --batch_size 2048 --n_layer 0 --embed_type ngcf --sample_rate 3 --sample_ratio 4 --eval_types valid test --eval_epoch 1 --early_stop 10 
```

### 由我们改进后的CPR

以下命令可以复现我们对CPR进行创新性的改进后，以 LightGCN 为骨干，在2个数据集（MovieLens, AmazonBook）上的结果：

```shell
python CPR.py --loss_type improved --dataset movielens_20m --lr 0.0001 --reg 0.001 --ks 20 --batch_size 2048 --n_layer 0 --embed_type lightgcn --sample_rate 3 --sample_ratio 3 --eval_types valid test --eval_epoch 1 --early_stop 10
```

```shell
python CPR.py --loss_type improved --dataset amazonbook --lr 0.001 --reg 0.001 --ks 20 --batch_size 2048 --n_layer 0 --embed_type lightgcn --sample_rate 3 --sample_ratio 4 --eval_types valid test --eval_epoch 1 --early_stop 10
```

### Baselines

代码还实现了一些 baseline 模型。以下命令可以复现这些 baseline 以 LightGCN 为骨干在两个数据集上的结果：


#### BPR

```shell
python BPR.py --dataset movielens_20m --lr 0.0001 --reg 0 --ks 20 --batch_size 2048 --n_layer 3 --embed_type lightgcn --eval_types valid test --eval_epoch 4 --early_stop 10 
```

```shell
python BPR.py --dataset amazonbook --lr 0.0001 --reg 0.00001 --ks 20 --batch_size 2048 --n_layer 3 --embed_type lightgcn --eval_types valid test --eval_epoch 4 --early_stop 10 
```

#### UBPR

```shell
python UBPR.py --dataset movielens_20m --lr 0.0001 --reg 0.001 --ks 20 --batch_size 2048 --n_layer 3 --embed_type lightgcn --ps_pow 0.8 --clip 0 --eval_types valid test --eval_epoch 4 --early_stop 10 
```

```shell
python UBPR.py --dataset amazonbook --lr 0.0001 --reg 0.0001 --ks 20 --batch_size 2048 --n_layer 3 --embed_type lightgcn --ps_pow 0.7 --clip 0 --eval_types valid test --eval_epoch 4 --early_stop 10 
```

#### DICE

```shell
python DICE.py --dataset movielens_20m --lr 0.0001 --reg 0.01 --ks 20 --batch_size 2048 --n_layer 3 --embed_type lightgcn --int_weight 9 --pop_weight 9 --dis_pen 0.0001 --margin 10 --margin_decay 0.9 --loss_decay 0.9 --eval_types valid test --eval_epoch 4 --early_stop 10 
```

```shell
python DICE.py --dataset amazonbook --lr 0.0001 --reg 0.01 --ks 20 --batch_size 2048 --n_layer 3 --embed_type lightgcn --int_weight 9 --pop_weight 9 --dis_pen 0.0001 --margin 40 --margin_decay 0.9 --loss_decay 0.9 --eval_types valid test --eval_epoch 4 --early_stop 10 
```

## Output

下面是一个输出日志的示例。这是以下命令的输出结果：

```shell
python CPR.py --dataset movielens_20m --lr 0.0001 --reg 0.001 --ks 20 --batch_size 2048 --n_layer 0 --embed_type lightgcn --sample_rate 3 --sample_ratio 3 --eval_types valid test --eval_epoch 4 --early_stop 10 --k_interact 4
```

```
...
Epoch 1 :   95.87836 s | loss = 0.69250 = 0.69249 + 0.00000
Epoch 2 :   96.82119 s | loss = 0.67755 = 0.67752 + 0.00003
Epoch 3 :   98.04564 s | loss = 0.63963 = 0.63955 + 0.00008
Epoch 4 :   98.59581 s | loss = 0.58808 = 0.58793 + 0.00016
============================================================================================================================================
[ valid set ]
---- Item ----
Recall    @20 :   0.14139
Precision @20 :   0.01831
NDCG      @20 :   0.07281
Rec       @20 :  20.00000
ARP       @20 :3142.82837
[ test set ]
---- Item ----
Recall    @20 :   0.15001
Precision @20 :   0.02895
NDCG      @20 :   0.08773
ARP       @20 :3258.51514
Evaluation :    1.71451 s
============================================================================================================================================
...
Early stopping triggered.
Best epoch: 280.
[ test set ]
---- Item ----
Recall    @20 :   0.18651
Precision @20 :   0.03730
NDCG      @20 :   0.11403
ARP       @20 :2334.42383
============================================================================================================================================
```
