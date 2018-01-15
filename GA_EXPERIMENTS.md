## IPGA with 2 bytes Xavier weight initialisation

## Empirical Settings

GA parameters elitism_rate: 0.1, mutation_rate: 0.2

#### MNIST Database from Tensorflow

**Search the best agent**

```bash
nohup python3 ga_main.py -d mb -m 1 --partial_dataset 0.1 --ip_structure 2 --elitism_rate 0.1 --mutation_rate 0.1,0.2 -s 30 -l 8 --max_generation 30 -e 10 -f 0 -g 1 --log_file=log/ipga_cnn_0001.log --gbest_file=log/ga_gbest_0001.pkl &
```

**Train the best agent**

```bash
nohup python3 ga_main.py -d mb -m 1 --ip_structure 2 -e 100 -f 0 -g 1 -o 1 --log_file=log/ipga_cnn_optimise_0001.log --gbest_file=log/ga_gbest_0001.pkl &
```

#### MDRBI(MNIST Digits rotated background images) dataset

**Search the best agent**

```bash
nohup python3 ga_main.py -d mdrbi -m 1 --partial_dataset 0.1 --ip_structure 2 --elitism_rate 0.1 --mutation_rate 0.1,0.2 -s 30 -l 8 --max_generation 30 -e 10 -f 0 -g 1 --log_file=log/ipga_cnn_0002.log --gbest_file=log/ga_gbest_0002.pkl &
```

**Train the best agent**

```bash
nohup python3 ga_main.py -d mdrbi -m 1 --ip_structure 2 -e 100 -f 0 -g 1 -o 1 --log_file=log/ipga_cnn_optimise_0002.log --gbest_file=log/ga_gbest_0002.pkl &
```

#### Convex dataset

**Search the best agent**

```bash
nohup python3 ga_main.py -d convex -m 1 --partial_dataset 0.1 --ip_structure 2 --elitism_rate 0.1 --mutation_rate 0.1,0.2 -s 30 -l 8 --max_generation 30 -e 10 -f 0 -g 1 --log_file=log/ipga_cnn_0003.log --gbest_file=log/ga_gbest_0003.pkl &
```

**Train the best agent**

```bash
nohup python3 ga_main.py -d convex -m 1 --ip_structure 2 -e 100 -f 0 -g 1 -o 1 --log_file=log/ipga_cnn_optimise_0003.log --gbest_file=log/ga_gbest_0003.pkl &
```