## IPDE with 2 bytes Xavier weight initialisation

## Empirical Settings

DE parameters f: 0.5, cr: 0.5

#### MNIST Database from Tensorflow

**Search the best agent**

```bash
nohup python3 de_main.py -d mb -m 1 --partial_dataset 0.1 --ip_structure 2 --f_weight 0.5 --cr 0.5 -s 30 -l 8 --max_generation 15 -e 10 -f 0 -g 1 --log_file=log/ipde_cnn_0001.log --gbest_file=log/de_gbest_0001.pkl &
```

**Train the best agent**

```bash
nohup python3 de_main.py -d mb -m 1 --ip_structure 2 -e 100 -f 0 -g 1 -o 1 --log_file=log/ipde_cnn_optimise_0001.log --gbest_file=log/de_gbest_0001.pkl &
```

#### MDRBI(MNIST Digits rotated background images) dataset

**Search the best agent**

```bash
nohup python3 de_main.py -d mdrbi -m 1 --partial_dataset 0.1 --ip_structure 2 --f_weight 0.5 --cr 0.5 -s 30 -l 8 --max_generation 15 -e 10 -f 0 -g 1 --log_file=log/ipde_cnn_0002.log --gbest_file=log/de_gbest_0002.pkl &
```

**Train the best agent**

```bash
nohup python3 de_main.py -d mdrbi -m 1 --ip_structure 2 -e 100 -f 0 -g 1 -o 1 --log_file=log/ipde_cnn_optimise_0002.log --gbest_file=log/de_gbest_0002.pkl &
```

#### Convex dataset

**Search the best agent**

```bash
nohup python3 de_main.py -d convex -m 1 --partial_dataset 0.1 --ip_structure 2 --f_weight 0.5 --cr 0.5 -s 30 -l 8 --max_generation 15 -e 10 -f 0 -g 1 --log_file=log/ipde_cnn_0003.log --gbest_file=log/de_gbest_0003.pkl &
```

**Train the best agent**

```bash
nohup python3 de_main.py -d convex -m 1 --ip_structure 2 -e 100 -f 0 -g 1 -o 1 --log_file=log/ipde_cnn_optimise_0003.log --gbest_file=log/de_gbest_0003.pkl &
```