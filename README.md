# Rearchitecting Classification Frameworks For Increased Robustness (YOPO Fork)
YOPO adversarial training experiments: [Brian Tang](https://byron123t.github.io/) and [Kassem Fawaz](https://kassemfawaz.com/)

[Varun Chandrasekaran](http://pages.cs.wisc.edu/~chandrasekaran/), [Brian Tang](https://byron123t.github.io/), [Nicholas Papernot](https://www.papernot.fr/), [Kassem Fawaz](https://kassemfawaz.com/), [Somesh Jha](http://pages.cs.wisc.edu/~jha/), [Xi Wu](http://andrewxiwu.github.io/)

Sample code for our paper, "Rearchitecting Classification Frameworks For Increased Robustness" [arXiv](https://arxiv.org/abs/1905.10900)

## Requirements
This code is tested with Python 3.5.2
Other required packages can be found in requirements.txt
Sample virtual environment commands:
```
python3 -m venv path_to_environment/
source path_to_environment/bin/activate
pip install -r requirements.txt
```

## How to run our code
- train.py -- Train a YOPO model
- config.py -- Contains hyperparameters
- dataset.py -- Data processing and equivalence classes
- eval.py -- Evaluates a model checkpoint
- eval_test_ids.py -- Does end to end evaluation of the hierarchy
- loss.py -- Contains loss function
- training_function.py -- Contains training functionality

You can change all the hyper-parameters in `config.py`. And change network in `network.py`
Actually code in above mentioned director is very **flexible** and can be easiliy modified. It can be used as a **template**. 

### YOPO training
Go to directory `experiments/CIFAR10/binary`
run `python train.py -d <which_gpu>`
change checkpoint filename and run `python eval_test_ids.py`

Go to directory `experiments/CIFAR10/4leaf`
run `python train.py -d <which_gpu>`
change checkpoint filename and run `python eval_test_ids.py`

Go to directory `experiments/CIFAR10/6leaf`
run `python train.py -d <which_gpu>`
change checkpoint filename and run `python eval_test_ids.py`

Go to directory `experiments/CIFAR10/full`
run `python train.py -d <which_gpu>`
run `python eval.py --resume <path_to_checkpoint>`

You can change all the hyper-parameters in `config.py`. And change network in `network.py`
Runing this code for the first time will dowload the dataset in `./experiments/CIFAR10/data/`, you can modify the path in `dataset.py`

## Miscellaneous
- Forked from this [repository](https://github.com/xiangchong1/3d-adv-pc)
- We leave integrating other datasets and generalizable code for creating and evaluating hierarchies as future work.
- Please open an issue or contact Brian Tang (byron123t@gmail.com) if there is any question.

## Cite
```
@article{chandrasekaran2019,
  title={Rearchitecting Classification Frameworks For Increased Robustness},
  author={Chandrasekaran, Varun and Tang, Brian and Papernot, Nicolas and Fawaz, Kassem and Jha, Somesh and Wu, Xi},
  journal={arXiv preprint arXiv:1905.10900},
  year={2019}
}
```
