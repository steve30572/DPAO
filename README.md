# DPAO

This is our PyTorch implementation of the paper:
>Dual Policy Learning for Aggregation Optimization in 
> Graph Neural Network-based Recommender Systems. 
> Heesoo Jung, Sangpil Kim and Hogun Park.<br> 
> In The Web Conference (WWW) 2023. <br>


The DPAO algorithms adaptively optimizes the number of GNN layers for each user and item by utilizing RL. <br>
Please check the [Paper](https://arxiv.org/abs/2302.10567) for more details.

### Environment Requirement

The code has been tested running under Python 3.9.15. The required packages are: <br>
- pytorch==1.12.0
- dgl-cu116
- numpy==1.23.4
- scikit-learn==1.2.1
- scipy==1.9.3
- pandas==1.5.3

### Reproducing Paper's Experiment

#### KG-based

###### Amazon-Book
```
cd KG
python main_DPAO.py --data_name amazon-book --conv_dim_list '[128, 64, 32, 16]' --mess_dropout '[0.1, 0.1, 0.1, 0.1]' --aggregation_type 'gcn' --evaluate_every 1 --lr 0.0001 --n_epoch -1
```


#### Non-KG-based

##### Gowalla
```
cd NonKG
python main.py --dataset Gowalla
```

##### MovieLens1M
``` 
cd NonKG
python main.py --dataset ml-1m
```
##### Amazon-Book
``` 
cd NonKG
python main.py --dataset amazon-book
```

### Citing
If you want to use our codes and papers in your research, please consider citing the paper:
```
    @inproceedings{DPAO,
    author={Heesoo Jung, Sangpil Kim and Hogun Park},
    title={Dual Policy Learning for Aggregation Opimization in Graph Neural Network-based Recommender Systems.},
    booktitle={Proceedings of the Web Conference},
    year={2023}
    }
```

### Acknowledgement
We refer to the code of [KGAT](https://github.com/LunaBlack/KGAT-pytorch/tree/kgat-dgl) and [NGCF](https://github.com/huangtinglin/NGCF-PyTorch). Thanks for their contribution
