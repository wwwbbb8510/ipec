### Overview

The IPPSO package is comprised of four parts/sub-packages - cnn(define CNN layers), data(provide datasets), ip(ip encoding and decoding) and pso(pso related methods).  

### Installation

The package is developed in Python 3 and there are several packages used - tensorflow, scipy and numpy, so in order to use the ISPSO package Python 3 and the two dependent packages have to be set up first. 

Once the environment is set up, the IPPSO package can be imported and called from any python code. 

However, to make it easier for debug it would be better to import the whole package into an IDE like pycharm. 

### Run your first program to perform IPPSO search

Go to the root folder of the repository

#### get help

```bash
python3 main.py -h
```

```text
usage: main.py [-h] [-d DATASET] [-m MODE] [-s POP_SIZE] [-l PARTICLE_LENGTH]
               [--max_steps MAX_STEPS] [-e TRAINING_EPOCH] [-g MAX_GPU]
               [-o OPTIMISE]

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        choose a dataset among mb, mdrbi, or convex
  -m MODE, --mode MODE  default:None, 1: production (load full data)
  -s POP_SIZE, --pop_size POP_SIZE
                        population size
  -l PARTICLE_LENGTH, --particle_length PARTICLE_LENGTH
                        particle max length
  --max_steps MAX_STEPS
                        max fly steps
  -e TRAINING_EPOCH, --training_epoch TRAINING_EPOCH
                        training epoch for the evaluation
  -f FIRST_GPU_ID, --first_gpu_id FIRST_GPU_ID
                        first gpu id
  -g MAX_GPU, --max_gpu MAX_GPU
                        max number of gpu
  -o OPTIMISE, --optimise OPTIMISE
                        optimise the learned CNN architecture. Default: None.
                        1: optimise; otherwise IPPSO search
  --log_file LOG_FILE   the path of log file
  --gbest_file GBEST_FILE
                        the path of gbest file
  --w W                 w parameter of PSO
  --c1 C1               c1 parameter of PSO
  --c2 C2               c2 parameter of PSO
  -v V_MAX, --v_max V_MAX
                        PSO max velocity used by velocity clamping
  -r REGULARISE, --regularise REGULARISE
                        weight regularisation hyper-parameter.
  --dropout DROPOUT     enable dropout and set dropout rate
  --ip_structure IP_STRUCTURE
                        IP structure. default: 5 bytes, 1: 3 bytes, 2: 2 bytes
                        with xavier weight initialisation
  --partial_dataset PARTIAL_DATASET
                        Use partial dataset for learning CNN architecture to
                        speed up the learning process.
```

#### run program in debug mode

```bash
python3 main.py -d mb -s 30 -l 10 --max_steps 30 -e 5 -f 0 -g 1
```

#### run program in production mode
 
```bash
python3 main.py -d mb -m 1 -s 30 -l 10 --max_steps 30 -e 5 -f 0 -g 1
``` 

#### run program in production mode in background
 
```bash
nohup python3 main.py -d mb -m 1 -s 30 -l 10 --max_steps 30 -e 5 -f 0 -g 1 --log_file=log/ippso_cnn.log --gbest_file=log/gbest.pkl &
```

After the program run, all the main steps can be checked in the log/ippso_cnn.log file and the global best particle will be persisted into log/gbest.pkl file.

### tweak PSO parameters in production mode in background

```bash
nohup python3 main.py -d mb -m 1 --w 0.1 --c1 0.02,0.1,0.1,0.1,0.1 --c2 0.02,0.1,0.1,0.1,0.1 -s 30 -l 10 --max_steps 30 -e 5 -f 0 -g 1 --log_file=log/ippso_cnn.log --gbest_file=log/gbest.pkl &
```

### Run your first program to optimise the learned CNN architecture

#### run program in production mode
 
```bash
python3 main.py -d mb -m 1 -e 30 -g 1 -o 1
```

#### run program in production mode in background
 
```bash
nohup python3 main.py -d mb -m 1 -e 30 -g 1 -o 1 &
```