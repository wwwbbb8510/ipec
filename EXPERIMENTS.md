### Empirical Settings 

PSO parameters w: 0.7298, c1: (1.49618,1.49618,1.49618,1.49618,1.49618), c2: (1.49618,1.49618,1.49618,1.49618,1.49618)

#### MNIST Basic dataset

**Search the best particle**

```bash
nohup python3 main.py -d mb -m 1 --w 0.7298 --c1 1.49618,1.49618,1.49618,1.49618,1.49618 --c2 1.49618,1.49618,1.49618,1.49618,1.49618 -s 30 -l 10 --max_steps 30 -e 10 -f 0 -g 1 --log_file=log/ippso_cnn_311.log --gbest_file=log/gbest_311.pkl &
```

**Search the best particle with velocity clamping**

```bash
nohup python3 main.py -d mb -m 1 --w 0.7298 --c1 1.49618,1.49618,1.49618,1.49618,1.49618 --c2 1.49618,1.49618,1.49618,1.49618,1.49618 -v 0.4,25.6,25.6,25.6,25.6 -s 30 -l 10 --max_steps 30 -e 10 -f 0 -g 1 --log_file=log/ippso_cnn_311.log --gbest_file=log/gbest_311.pkl &
```

**Train the best particle**

```bash
nohup python3 main.py -d mb -m 1 -e 100 -f 0 -g 1 -o 1 --log_file=log/ippso_cnn_optimise_311.log --gbest_file=log/gbest_311.pkl &
```

#### MDRBI(MNIST Digits rotated background images) dataset

**Search the best particle**

```bash
nohup python3 main.py -d mdrbi -m 1 --w 0.7298 --c1 1.49618,1.49618,1.49618,1.49618,1.49618 --c2 1.49618,1.49618,1.49618,1.49618,1.49618 -s 30 -l 15 --max_steps 30 -e 10 -f 0 -g 1 --log_file=log/ippso_cnn_305.log --gbest_file=log/gbest_305.pkl &
```

**Search the best particle with velocity clamping**

```bash
nohup python3 main.py -d mdrbi -m 1 --w 0.7298 --c1 1.49618,1.49618,1.49618,1.49618,1.49618 --c2 1.49618,1.49618,1.49618,1.49618,1.49618 -v 0.4,25.6,25.6,25.6,25.6 -s 30 -l 15 --max_steps 30 -e 10 -f 0 -g 1 --log_file=log/ippso_cnn_305.log --gbest_file=log/gbest_305.pkl &
```

**Train the best particle**

```bash
nohup python3 main.py -d mdrbi -m 1 -e 100 -f 0 -g 1 -o 1 --log_file=log/ippso_cnn_optimise_305.log --gbest_file=log/gbest_305.pkl &
```

#### Convex dataset

**Search the best particle**

```bash
nohup python3 main.py -d convex -m 1 --w 0.7298 --c1 1.49618,1.49618,1.49618,1.49618,1.49618 --c2 1.49618,1.49618,1.49618,1.49618,1.49618 -s 30 -l 15 --max_steps 30 -e 10 -f 0 -g 1 --log_file=log/ippso_cnn_405.log --gbest_file=log/gbest_405.pkl &
```

**Search the best particle with velocity clamping**

```bash
nohup python3 main.py -d convex -m 1 --w 0.7298 --c1 1.49618,1.49618,1.49618,1.49618,1.49618 --c2 1.49618,1.49618,1.49618,1.49618,1.49618 -v 0.4,25.6,25.6,25.6,25.6 -s 30 -l 15 --max_steps 30 -e 10 -f 0 -g 1 --log_file=log/ippso_cnn_405.log --gbest_file=log/gbest_405.pkl &
```

**Train the best particle**

```bash
nohup python3 main.py -d convex -m 1 -e 100 -f 0 -g 1 -o 1 --log_file=log/ippso_cnn_optimise_405.log --gbest_file=log/gbest_405.pkl &
```