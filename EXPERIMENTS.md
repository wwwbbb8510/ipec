### MNIST Basic dataset

#### Empirical Settings 

PSO parameters w: 0.7298, c1: (0.0292,1.49618,1.49618,1.49618,1.49618), c2: (0.0292,1.49618,1.49618,1.49618,1.49618)

```bash
nohup python3 main.py -d mb -m 1 --w 0.7298 --c1 0.0292,1.49618,1.49618,1.49618,1.49618 --c2 0.0292,1.49618,1.49618,1.49618,1.49618 -s 30 -l 15 --max_steps 30 -e 10 -f 0 -g 1 --log_file=log/ippso_cnn_200.log --gbest_file=log/gbest_200.pkl &
```

#### 