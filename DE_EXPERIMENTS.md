## IPDE with 2 bytes Xavier weight initialisation

## Empirical Settings

DE parameters f: 0.5, cr: 0.5

#### MNIST Database from Tensorflow

**Search the best particle**

```bash
nohup python3 main.py -m 1 --ip_structure 2 --f_weight 0.5 --cr 0.5 -s 30 -l 8 --max_generation 15 -e 10 -f 0 -g 1 --log_file=log/ipde_cnn_0001.log --gbest_file=log/de_gbest_0001.pkl &
```

**Search the best particle with velocity clamping**

```bash
nohup python3 main.py -m 1 --ip_structure 2 --f_weight 0.5 --cr 0.5 -v 4,25.6 -s 30 -l 8 --max_generation 15 -e 10 -f 0 -g 1 --log_file=log/ipde_cnn_0001.log --gbest_file=log/de_gbest_0001.pkl &
```

**Train the best particle**

```bash
nohup python3 main.py -m 1 --ip_structure 2 -e 100 -f 0 -g 1 -o 1 --log_file=log/ipde_cnn_optimise_0001.log --gbest_file=log/de_gbest_0001.pkl &