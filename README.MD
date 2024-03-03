## Requirements

- Python == 3.7.13
- numpy == 1.21.5
- PyTorch == 1.11.0
- scikit-learn == 1.0.2
- transformers == 4.19.2
- numba == 0.56.2
- glove.6B.300d


### Conduct experiments on your own data
In [HiAGM](https://github.com/Alibaba-NLP/HiAGM), an additional step is required to count the prior probabilities between parent and child labels by running `python taxonompy.py`, `python helper/hiearchy_tree_statistic.py`. HiTIN only requires unweighted adjacency matrix of label hierarchies but we still retain this property and save the statistics in `data/DATASET_prob.json` as we also implement baseline methods including TextRCNN, BERT-based HiAGM. 

If you tend to evaluate these methods on your own dataset, please make sure to organize your data in the following format:
```
{
    "doc_label": ["Computer", "MachineLearning", "DeepLearning", "Neuro", "ComputationalNeuro"],
    "doc_token": ["I", "love", "deep", "learning"],
    "doc_keyword": ["deep learning"],
    "doc_topic": ["AI", "Machine learning"]
}

where "doc_keyword" and "doc_topic" are optional.
```
then, replace the label name with your dataset's in line143~146 of `helper/hierarchy_tree_statistics.py` and run:
```shell
python helper/hierarchy_tree_statistic.py config/~~
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

```shell
python train.py -cfg config/tin-custom-roberta.json ~~
```