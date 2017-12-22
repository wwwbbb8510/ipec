### Overview

The IPPSO package is comprised of four parts/sub-packages - cnn(define CNN layers), data(provide datasets), ip(ip encoding and decoding) and pso(pso related methods).  

### Installation

The package is developed in Python 3 and there are several packages used - tensorflow, scipy and numpy, so in order to use the ISPSO package Python 3 and the two dependent packages have to be set up first. 

Once the environment is set up, the IPPSO package can be imported and called from any python code. 

However, to make it easier for debug it would be better to import the whole package into an IDE like pycharm. 

### Run your first program

Go to the root folder of the repository

#### get help

```bash
python main.py -h
```

#### run program in debug mode

```bash
python main.py -d mb -s 30 -l 15 --max_steps 30
```

#### run program in production mode
 
```bash
python main.py -d mb -m 1 -s 30 -l 15 --max_steps 30
``` 

After the program run, all the main steps can be checked in the log/ippso_cnn.log file and the global best particle will be persisted into log/gbest.pkl file.