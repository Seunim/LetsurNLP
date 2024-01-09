# HiTIN: Hierarchy-aware Tree Isomorphism Network for Hierarchical Text Classification

Official implementation for ACL 2023 accepted paper "HiTIN: Hierarchy-aware Tree Isomorphism Network for Hierarchical Text Classification" . [[arXiv](https://arxiv.org/abs/2305.15182)][[pdf](https://arxiv.org/pdf/2305.15182.pdf)][[bilibili](https://www.bilibili.com/video/BV1vL411i7uY/?share_source=copy_web&vd_source=a9cc6ff9a8cf3c92bf2375da5b56a007)]

## Requirements

!pip install transformers

## Data preparation

text_total.json을 data/에 놓고
LetsurNLP directory에서 
```shell
python taxonomy.py -tp sum
python hierarchy_tree_statistics.py config/tin-custom-roberta.json
```

## Train
The default parameters are not the best performing-hyper-parameters used to reproduce our results in the paper. Hyper-parameters need to be specified through the commandline arguments. Please refer to our paper for the details of how we set the hyper-parameters.

To learn hyperparameters to be specified, please see: 
```
python train.py [-h] -cfg CONFIG_FILE [-b BATCH_SIZE] [-lr LEARNING_RATE]
                [-l2 L2RATE] [-p] [-k TREE_DEPTH] [-lm NUM_MLP_LAYERS]
                [-hd HIDDEN_DIM] [-fd FINAL_DROPOUT] [-tp {root,sum,avg,max}]
                [-hp HIERAR_PENALTY] [-ct CLASSIFICATION_THRESHOLD]
                [--log_dir LOG_DIR] [--ckpt_dir CKPT_DIR]
                [--begin_time BEGIN_TIME]

optional arguments:
  -h, --help            show this help message and exit
  -cfg CONFIG_FILE, --config_file CONFIG_FILE
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        input batch size for training (default: 32)
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate (default: 0.001)
  -l2 L2RATE, --l2rate L2RATE
                        L2 penalty lambda (default: 0.01)
  -p, --load_pretrained
  -k TREE_DEPTH, --tree_depth TREE_DEPTH
                        The depth of coding tree to be constructed by CIRCA
                        (default: 2)
  -lm NUM_MLP_LAYERS, --num_mlp_layers NUM_MLP_LAYERS
                        Number of layers for MLP EXCLUDING the input one
                        (default: 2). 1 means linear model.
  -hd HIDDEN_DIM, --hidden_dim HIDDEN_DIM
                        Number of hidden units for HiTIN layer (default: 512)
  -fd FINAL_DROPOUT, --final_dropout FINAL_DROPOUT
                        Dropout rate for HiTIN layer (default: 0.5)
  -tp {root,sum,avg,max}, --tree_pooling_type {root,sum,avg,max}
                        Pool strategy for the whole tree in Eq.11. Could be
                        chosen from {root, sum, avg, max}.
  -hp HIERAR_PENALTY, --hierar_penalty HIERAR_PENALTY
                        The weight for L^R in Eq.14 (default: 1e-6).
  -ct CLASSIFICATION_THRESHOLD, --classification_threshold CLASSIFICATION_THRESHOLD
                        Threshold of binary classification. (default: 0.5)
  --log_dir LOG_DIR     Path to save log files (default: log).
  --ckpt_dir CKPT_DIR   Path to save checkpoints (default: ckpt).
  --begin_time BEGIN_TIME
                        The beginning time of a run, which prefixes the name
                        of log files.
```

We provide a lot of config files in `./config`. 

**Before running, the last thing to do is modify the `YOUR_DATA_DIR`, `YOUR_BERT_DIR` in the json file.**

An example of training HiTIN on RCV1 with **TextRCNN** as the text encoder:
```shell
python train.py -cfg config/tin-rcv1-v2.json -k 2 -b 64 -hd 512 -lr 1e-4 -tp sum
```

An example of training HiTIN on WOS with **BERT** as the text encoder:
```shell
python train.py -cfg config/tin-wos-bert.json -k 2 -b 12 -hd 768 -lr 1e-4 -tp sum
```

## Citation
If you found the provided code with our paper useful in your work, please **star** this repo and **cite** our paper!
```
@inproceedings{zhu-etal-2023-hitin,
    title = "{H}i{TIN}: Hierarchy-aware Tree Isomorphism Network for Hierarchical Text Classification",
    author = "Zhu, He  and
      Zhang, Chong  and
      Huang, Junjie  and
      Wu, Junran  and
      Xu, Ke",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.432",
    pages = "7809--7821",
}
```
