# CMPRY

### Files in the folder

+ ``data``
  + test: contain 860 training data and 172 test data
  + strict_test: contain 700 training data and 32 test data
+ ``pretrain`` : codes to pretrain CMPNN and save pretrained model

+ ``src``:  codes of CMPNN

### Running the code

+ test using trained model

```bash
python test.py --dataset test

#or test on strict_test set
python test.py --dataset strict_test
```

+ train the model

```bash
#train model and test 
python run_gnn.py --dataset test

#or train model on strict_test set
python run_gnn.py --dataset strict_test --decay 0.02
```

+ running environment

```bash
conda env create -f environment.yaml
```

