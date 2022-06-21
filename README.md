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
python main.py

#or train model and test on strict_test set
python main_strict_test.py
```

+ running environment

```bash
conda env create -f environment.yaml
#or
pip install -r requirements.txt
```

