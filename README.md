### Overview

The IPPSO package is comprised of four parts/sub-packages - cnn(define CNN layers), data(provide datasets), ip(ip encoding and decoding) and pso(pso related methods).  

### Installation

The package is developed in Python 3 and there are two main packages used - tensorflow and numpy, so in order to use the ISPSO package Python 3 and the two dependent packages have to be set up first. 

Once the environment is set up, the IPPSO package can be imported and called from any python code. 

However, to make it easier for debug it would be better to import the whole package into an IDE like pycharm. 

### Run your first program

Go to the root folder of the repository

```bash
python main.py
``` 

After the program run, all the main steps can be checked in the log/ippso_cnn.log file and the global best particle will be persisted into log/gbest.pkl file.