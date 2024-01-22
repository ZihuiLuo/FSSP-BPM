# FSSP-BPM
Paper "Flow Shop Scheduling with Batch Processing Machines via Deep Reinforcement Learning for Industrial Internet of Things"


# About the baseline file
The baseline file contains the comparison algorithms in the paper, including PDRs (FIFO, LIFO, LPT, SPT), ACO, GA, TS, and DR_G. And DRL_IG you can read the paper "Scheduling optimization for flow-shop based on deep reinforcement learning and iterative greedy method" (DOI: 10.13195/j.kzyjc.2020.0608).

# About the DRL file:
## Dependencies
* Numpy
* [tensorflow](https://www.tensorflow.org/)>=1.2

## How to Run
### Train
By default, the code is running in the training mode on a single GPU. For running the code, one can use the following command:
```bash
python main.py
```

It is possible to add other config parameters like:
```bash
python main.py --task=vrp100 --gpu=0 --n_glimpses=1 --use_tanh=False 
```
There is a full list of all configs in the ``config.py`` file. Also, task-specific parameters are available in ``task_specific_params.py``
### Inference
For running the trained model for inference, it is possible to turn off the training mode. For this, you need to specify the directory of the trained model, otherwise, a random model will be used for decoding:
```bash
python main.py --task=vrp100 --is_train=False --model_dir=./path_to_your_saved_checkpoint
```
The default inference is run in batch mode, meaning that all testing instances are fed simultaneously. It is also possible to do inference in a single mode, which means that we decode instances one by one. The latter case is used for reporting the runtimes and it will display detailed reports. For running the inference with single mode, you can try:
```bash
python main.py --task=vrp100 --is_train=False --infer_type=single --model_dir=./path_to_your_saved_checkpoint
```
### Logs
All logs are stored in the ``result.txt`` file stored in the ``./logs/task_date_time`` directory.
## Sample FSP scheduling solution

## Acknowledgements
Thanks to [pemami4911/neural-combinatorial-rl-pytorch](https://github.com/pemami4911/neural-combinatorial-rl-pytorch) and the paper "Reinforcement Learning for Solving the Vehicle Routing Problem" for getting the idea of restructuring the code.
