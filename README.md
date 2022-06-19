# CMPRY

### Files in the folder

+ ``data``
  + test: contain 860 training data and 182 test data
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
python run_gnn.py --test 1 --dataset test --batch_size 128

#or train model on strict_test set
python run_gnn.py --test 1 --dataset strict_test
```

