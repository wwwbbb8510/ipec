### Empirical Settings 

PSO parameters w: 0.7298, c1: (1.49618,1.49618,1.49618,1.49618,1.49618), c2: (1.49618,1.49618,1.49618,1.49618,1.49618)

#### MNIST Basic dataset

**Search the best particle**

```bash
nohup python3 main.py -d mb -m 1 --w 0.7298 --c1 1.49618,1.49618,1.49618,1.49618,1.49618 --c2 1.49618,1.49618,1.49618,1.49618,1.49618 -s 30 -l 8 --max_steps 30 -e 10 -f 0 -g 1 --log_file=log/ippso_cnn_311.log --gbest_file=log/gbest_311.pkl &
```

**Search the best particle with velocity clamping**

```bash
nohup python3 main.py -d mb -m 1 --w 0.7298 --c1 1.49618,1.49618,1.49618,1.49618,1.49618 --c2 1.49618,1.49618,1.49618,1.49618,1.49618 -v 0.4,25.6,25.6,25.6,25.6 -s 30 -l 8 --max_steps 30 -e 10 -f 0 -g 1 --log_file=log/ippso_cnn_311.log --gbest_file=log/gbest_311.pkl &
```

**Train the best particle**

```bash
nohup python3 main.py -d mb -m 1 -e 100 -f 0 -g 1 -o 1 --log_file=log/ippso_cnn_optimise_311.log --gbest_file=log/gbest_311.pkl &
```

#### MDRBI(MNIST Digits rotated background images) dataset

**Search the best particle**

```bash
nohup python3 main.py -d mdrbi -m 1 --w 0.7298 --c1 1.49618,1.49618,1.49618,1.49618,1.49618 --c2 1.49618,1.49618,1.49618,1.49618,1.49618 -s 30 -l 8 --max_steps 30 -e 10 -r 0.005 -f 0 -g 1 --log_file=log/ippso_cnn_306.log --gbest_file=log/gbest_306.pkl &
```

**Search the best particle with velocity clamping**

```bash
nohup python3 main.py -d mdrbi -m 1 --w 0.7298 --c1 1.49618,1.49618,1.49618,1.49618,1.49618 --c2 1.49618,1.49618,1.49618,1.49618,1.49618 -v 0.4,25.6,25.6,25.6,25.6 -s 30 -l 8 --max_steps 30 -e 10 -r 0.005 -f 0 -g 1 --log_file=log/ippso_cnn_306.log --gbest_file=log/gbest_306.pkl &
```

**Train the best particle**

```bash
nohup python3 main.py -d mdrbi -m 1 -e 100 -r 0.005 -f 0 -g 1 -o 1 --log_file=log/ippso_cnn_optimise_306.log --gbest_file=log/gbest_306.pkl &
```

#### Convex dataset

**Search the best particle**

```bash
nohup python3 main.py -d convex -c 2 -m 1 --w 0.7298 --c1 1.49618,1.49618,1.49618,1.49618,1.49618 --c2 1.49618,1.49618,1.49618,1.49618,1.49618 -s 30 -l 8 --max_steps 30 -e 10 -f 0 -g 1 --log_file=log/ippso_cnn_406.log --gbest_file=log/gbest_406.pkl &
```

**Search the best particle with velocity clamping**

```bash
nohup python3 main.py -d convex -c 2 -m 1 --w 0.7298 --c1 1.49618,1.49618,1.49618,1.49618,1.49618 --c2 1.49618,1.49618,1.49618,1.49618,1.49618 -v 0.4,25.6,25.6,25.6,25.6 -s 30 -l 8 --max_steps 30 -e 10 -f 0 -g 1 --log_file=log/ippso_cnn_406.log --gbest_file=log/gbest_406.pkl &
```

**Train the best particle**

```bash
nohup python3 main.py -d convex -c 2 -m 1 -e 100 -f 0 -g 1 -o 1 --log_file=log/ippso_cnn_optimise_406.log --gbest_file=log/gbest_406.pkl &
```