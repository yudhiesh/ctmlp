# Continuous Testing of Machine Learning Projects

Testing software is vital to ensure that code behaves as expected. In Machine Learning projects, testing is not as widely common as normal software testing. The aim of this talk is to give a brief overview on unit testing and to show how a Data Scientist/Machine Learning Engineer can implement it in a modern Machine Learning Development Lifecycle along with DevOps principles such as CI/CD.

## Badges

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
![Tests](https://github.com/yudhiesh/ctmlp/actions/workflows/main.yml/badge.svg)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)

## Run Locally

Clone the project

```bash
git clone https://github.com/yudhiesh/ctmlp
```

Create the conda environment

```bash
conda create --name ctmlp python=3.7
conda activate ctmlp
```

Install dependencies

```bash
pip install -r requirements.txt
```

Train a model

```bash
python src/models/train_model.py --train_path="./data/raw/train.csv" --test_path="./data/raw/test.csv"
```

## Running Tests

To run tests, run the following command

```bash
pytest --no-header -v
```

## Documentation

```
├── LICENSE
├── README.md
├── conftest.py             <- shares fixtures for test to all test
├── data                    <- data used
│   └── raw
│       ├── data_description.txt
│       ├── test.csv        <- test data
│       └── train.csv       <- train data
├── models
│   └── model.pkl           <- saved model that was trained
├── pytest.ini              <- configurations that are used for tests
├── requirements.txt        <- dependencies
├── setup.cfg               <- configures the behavior of the various setup commands for the project
├── src
│   ├── __init__.py
│   └── models
│       ├── __init__.py
│       └── train_model.py  <- script to train the model
├── test_score.json         <- json of the model metrics from training
└── tests
    ├── helpers
    │   ├── __init__.py
    │   └── utils.py        <- helper methods used in test
    └── test_post_train.py  <- contains post training test
```

### Tests

```python
# pre-train tests
# located at src/models/train_model.py

is_data_leaking()                    # checks if there is data leakage detected
is_overfitting_batch                 # checks if the model is able to overfit a single batch of data

# post-train tests
# located at tests/test_post_train.py

test_invariance_tests()              # checks for small perturbations that should not impact the models predictions
test_directional_expectation_tests() # checks for small perturbations that should impact the model
test_model_inference_times()         # check that the models inference speed at the 99th percentile is acceptable
test_model_metric                    # check that the models metric is below a set score
```

## Related

Here are some resources I used when coming up with this talk

- [How to Test Machine Learning Code and Systems](https://eugeneyan.com/writing/testing-ml/)
- [Effective testing for machine learning systems](https://www.jeremyjordan.me/testing-ml/)
- [Automated Testing for Machine Learing](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjJveLKiML0AhWfSGwGHbTYDGcQwqsBegQIBRAB&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DbSXUJRnQPPo&usg=AOvVaw3pv3kp6stu2UcgfO0BdQrW)

## Optimizations

- Decouple the model definition from the training code to ensure more flexibility
- Add in more test cases
- use DVC to version data as data in the real world would be too big to include inside of a repository
