## Quick commands

**batch n runs**

```bash
# evolve
nohup bash ./run_m_runs.sh -r 5 -p 22 -e 1 -f 0 &
# optimise
nohup bash ./run_m_runs.sh -r 5 -p 22 -e 2 &
```

**batch running on multiple tasks through bash scripts**

```bash
# batch train CNNs
nohup bash ./run_all.sh -g 1 -i 0002 -e 1 &
# batch optimise final models
nohup bash ./run_all.sh -g 1 -i 0002 -e 2 &
```

**bach script for re-evolve**

```bash
nohup bash ./run_re_evolve.sh -g 0 -i 1003 -d mbi &
nohup bash ./run_re_evolve.sh -g 1 -i 1003 -d mrd &
nohup bash ./run_re_evolve.sh -g 1 -i 1003 -d mrb &
```

**go to grid folder**

```bash
/vol/grid-solar/sgeusers/wangbin
```

**check disk usage**

```bash
du -d1 | sort -n
```

**copy datasets to cuda server**

```bash
scp -r datasets/* wangbin@bats.ecs.vuw.ac.nz:~/code/psocnn-comp440/datasets/
```

**Extract gbest in log files**

```bash
grep 'fitness of gbest' log/ippso_cnn_3132.log log/ippso_cnn_4032.log log/ippso_cnn_3032.log
grep 'fitness of gbest' log/ippso_cnn_3132.log
grep 'fitness of gbest' log/ippso_cnn_4032.log
grep 'fitness of gbest' log/ippso_cnn_3032.log
```

**Monitor training the final gbest**

```bash
tail -f log/ippso_cnn_optimise_3132.log
tail -f log/ippso_cnn_optimise_3032.log
tail -f log/ippso_cnn_optimise_4032.log
```

**batch train final CNNs**

```bash
# convex
python3 main.py -d convex -m 1 --ip_structure 2 -e 100 -f 0 -g 1 -o 1 --log_file=log/ippso_cnn_optimise_convex_13002.log --gbest_file=log/gbest_convex_13002.pkl
python3 main.py -d convex -m 1 --ip_structure 2 -e 100 -f 1 -g 1 -o 1 --log_file=log/ippso_cnn_optimise_convex_11002.log --gbest_file=log/gbest_convex_11002.pkl
python3 main.py -d convex -m 1 --ip_structure 2 -e 100 -f 1 -g 1 -o 1 --log_file=log/ippso_cnn_optimise_convex_12002.log --gbest_file=log/gbest_convex_12002.pkl
# mb
python3 main.py -d mb -m 1 --ip_structure 2 -e 100 -f 0 -g 1 -o 1 --log_file=log/ippso_cnn_optimise_mb_17002.log --gbest_file=log/gbest_mb_17002.pkl
# mdrbi
python3 main.py -d mdrbi -m 1 --ip_structure 2 -e 100 -f 1 -g 1 -o 1 --log_file=log/ippso_cnn_optimise_mdrbi_12002.log --gbest_file=log/gbest_mdrbi_12002.pkl
python3 main.py -d mdrbi -m 1 --ip_structure 2 -e 100 -f 1 -g 1 -o 1 --log_file=log/ippso_cnn_optimise_mdrbi_17002.log --gbest_file=log/gbest_mdrbi_17002.pkl
```

## IPPSO with 2 bytes Xavier weight initialisation

## Empirical Settings

PSO parameters w: 0.7298, c1: (1.49618,1.49618), c2: (1.49618,1.49618)

#### MNIST Database from Tensorflow

**Search the best particle**

```bash
nohup python3 main.py -m 1 --ip_structure 2 --w 0.7298 --c1 1.49618,1.49618 --c2 1.49618,1.49618 -s 30 -l 8 --max_steps 15 -e 10 -f 0 -g 1 --log_file=log/ippso_cnn_2032.log --gbest_file=log/gbest_2032.pkl &
```

**Search the best particle with velocity clamping**

```bash
nohup python3 main.py -m 1 --ip_structure 2 --w 0.7298 --c1 1.49618,1.49618 --c2 1.49618,1.49618 -v 4,25.6 -s 30 -l 8 --max_steps 15 -e 10 -f 0 -g 1 --log_file=log/ippso_cnn_2032.log --gbest_file=log/gbest_2032.pkl &
```

**Train the best particle**

```bash
nohup python3 main.py -m 1 --ip_structure 2 -e 100 -f 0 -g 1 -o 1 --log_file=log/ippso_cnn_optimise_2032.log --gbest_file=log/gbest_2032.pkl &
```

#### MNIST Basic dataset

**Search the best particle**

```bash
nohup python3 main.py -d mb -m 1 --ip_structure 2 --w 0.7298 --c1 1.49618,1.49618 --c2 1.49618,1.49618 -s 30 -l 8 --max_steps 15 -e 10 -f 0 -g 1 --log_file=log/ippso_cnn_3135.log --gbest_file=log/gbest_3135.pkl &
```

**Search the best particle with velocity clamping**

```bash
nohup python3 main.py -d mb -m 1 --partial_dataset 0.1 --ip_structure 2 --w 0.7298 --c1 1.49618,1.49618 --c2 1.49618,1.49618 -v 4,25.6 -s 30 -l 8 --max_steps 30 -e 10 -f 0 -g 1 --log_file=log/ippso_cnn_3135.log --gbest_file=log/gbest_3135.pkl &
```

**Train the best particle**

```bash
nohup python3 main.py -d mb -m 1 --ip_structure 2 -e 100 -f 0 -g 1 -o 1 --log_file=log/ippso_cnn_optimise_3135.log --gbest_file=log/gbest_3135.pkl &
```

#### MDRBI(MNIST Digits rotated background images) dataset

**Search the best particle**

```bash
nohup python3 main.py -d mdrbi -m 1 --ip_structure 2 --w 0.7298 --c1 1.49618,1.49618 --c2 1.49618,1.49618 -s 30 -l 8 --max_steps 15 -e 10 -f 0 -g 1 --log_file=log/ippso_cnn_3035.log --gbest_file=log/gbest_3035.pkl &
```

**Search the best particle with velocity clamping**

```bash
nohup python3 main.py -d mdrbi -m 1 --partial_dataset 0.1 --ip_structure 2 --w 0.7298 --c1 1.49618,1.49618 --c2 1.49618,1.49618 -v 4,25.6 -s 30 -l 8 --max_steps 30 -e 10 -f 0 -g 1 --log_file=log/ippso_cnn_3035.log --gbest_file=log/gbest_3035.pkl &
```

**Train the best particle**

```bash
nohup python3 main.py -d mdrbi -m 1 --ip_structure 2 -e 100 -f 0 -g 1 -o 1 --log_file=log/ippso_cnn_optimise_3035.log --gbest_file=log/gbest_3035.pkl &
```

#### Convex dataset

**Search the best particle**

```bash
nohup python3 main.py -d convex -c 2 -m 1 --ip_structure 2 --w 0.7298 --c1 1.49618,1.49618 --c2 1.49618,1.49618 -s 30 -l 8 --max_steps 15 -e 10 -f 0 -g 1 --log_file=log/ippso_cnn_4035.log --gbest_file=log/gbest_4035.pkl &
```

**Search the best particle with velocity clamping**

```bash
nohup python3 main.py -d convex -c 2 -m 1 --partial_dataset 0.15 --ip_structure 2 --w 0.7298 --c1 1.49618,1.49618 --c2 1.49618,1.49618 -v 4,25.6 -s 30 -l 8 --max_steps 30 -e 10 -f 0 -g 1 --log_file=log/ippso_cnn_4035.log --gbest_file=log/gbest_4035.pkl &
```

**Train the best particle**

```bash
nohup python3 main.py -d convex -c 2 -m 1 --ip_structure 2 -e 100 -r 0.5 --dropout 0.5 -f 0 -g 1 -o 1 --log_file=log/ippso_cnn_optimise_4035.log --gbest_file=log/gbest_4035.pkl &
```


## IPPSO with 3 bytes IP

## Empirical Settings

PSO parameters w: 0.7298, c1: (1.49618,1.49618,1.49618), c2: (1.49618,1.49618,1.49618)

#### MNIST Database from Tensorflow

**Search the best particle**

```bash
nohup python3 main.py -m 1 --ip_structure 1 --w 0.7298 --c1 1.49618,1.49618,1.49618 --c2 1.49618,1.49618,1.49618 -s 30 -l 8 --max_steps 30 -e 10 -f 0 -g 1 --log_file=log/ippso_cnn_2033.log --gbest_file=log/gbest_2033.pkl &
```

**Search the best particle with velocity clamping**

```bash
nohup python3 main.py -m 1 --ip_structure 1 --w 0.7298 --c1 1.49618,1.49618,1.49618 --c2 1.49618,1.49618,1.49618 -v 4,25.6,25.6 -s 30 -l 8 --max_steps 30 -e 10 -f 0 -g 1 --log_file=log/ippso_cnn_2033.log --gbest_file=log/gbest_2033.pkl &
```

**Train the best particle**

```bash
nohup python3 main.py -m 1 --ip_structure 1 -e 100 -f 0 -g 1 -o 1 --log_file=log/ippso_cnn_optimise_2033.log --gbest_file=log/gbest_2033.pkl &
```

#### MNIST Basic dataset

**Search the best particle**

```bash
nohup python3 main.py -d mb -m 1 --ip_structure 1 --w 0.7298 --c1 1.49618,1.49618,1.49618 --c2 1.49618,1.49618,1.49618 -s 30 -l 8 --max_steps 30 -e 10 -f 0 -g 1 --log_file=log/ippso_cnn_3133.log --gbest_file=log/gbest_3133.pkl &
```

**Search the best particle with velocity clamping**

```bash
nohup python3 main.py -d mb -m 1 --ip_structure 1 --w 0.7298 --c1 1.49618,1.49618,1.49618 --c2 1.49618,1.49618,1.49618 -v 4,25.6,25.6 -s 30 -l 8 --max_steps 30 -e 10 -f 0 -g 1 --log_file=log/ippso_cnn_3133.log --gbest_file=log/gbest_3133.pkl &
```

**Train the best particle**

```bash
nohup python3 main.py -d mb -m 1 --ip_structure 1 -e 100 -f 0 -g 1 -o 1 --log_file=log/ippso_cnn_optimise_3133.log --gbest_file=log/gbest_3133.pkl &
```

#### MDRBI(MNIST Digits rotated background images) dataset

**Search the best particle**

```bash
nohup python3 main.py -d mdrbi -m 1 --ip_structure 1 --w 0.7298 --c1 1.49618,1.49618,1.49618 --c2 1.49618,1.49618,1.49618 -s 30 -l 8 --max_steps 30 -e 10 -r 0.01 -f 0 -g 1 --log_file=log/ippso_cnn_3033.log --gbest_file=log/gbest_3033.pkl &
```

**Search the best particle with velocity clamping**

```bash
nohup python3 main.py -d mdrbi -m 1 --ip_structure 1 --w 0.7298 --c1 1.49618,1.49618,1.49618 --c2 1.49618,1.49618,1.49618 -v 4,25.6,25.6 -s 30 -l 8 --max_steps 30 -e 10 -r 0.01 -f 0 -g 1 --log_file=log/ippso_cnn_3033.log --gbest_file=log/gbest_3033.pkl &
```

**Train the best particle**

```bash
nohup python3 main.py -d mdrbi -m 1 --ip_structure 1 -e 100 -r 0.01 --dropout 0.5 -f 0 -g 1 -o 1 --log_file=log/ippso_cnn_optimise_3033.log --gbest_file=log/gbest_3033.pkl &
```

#### Convex dataset

**Search the best particle**

```bash
nohup python3 main.py -d convex -c 2 -m 1 --ip_structure 1 --w 0.7298 --c1 1.49618,1.49618,1.49618 --c2 1.49618,1.49618,1.49618 -s 30 -l 8 --max_steps 30 -e 10 -f 0 -g 1 --log_file=log/ippso_cnn_4031.log --gbest_file=log/gbest_4031.pkl &
```

**Search the best particle with velocity clamping**

```bash
nohup python3 main.py -d convex -c 2 -m 1 --ip_structure 1 --w 0.7298 --c1 1.49618,1.49618,1.49618 --c2 1.49618,1.49618,1.49618 -v 4,25.6,25.6 -s 30 -l 8 --max_steps 30 -e 10 -f 0 -g 1 --log_file=log/ippso_cnn_4031.log --gbest_file=log/gbest_4031.pkl &
```

**Train the best particle**

```bash
nohup python3 main.py -d convex -c 2 -m 1 --ip_structure 1 -e 100 -r 0.5 --dropout 0.5 -f 0 -g 1 -o 1 --log_file=log/ippso_cnn_optimise_4031.log --gbest_file=log/gbest_4031.pkl &
```

## IPPSO with 5 bytes IP

### Empirical Settings

PSO parameters w: 0.7298, c1: (1.49618,1.49618,1.49618,1.49618,1.49618), c2: (1.49618,1.49618,1.49618,1.49618,1.49618)

#### MNIST Database from Tensorflow

**Search the best particle**

```bash
nohup python3 main.py -m 1 --w 0.7298 --c1 1.49618,1.49618,1.49618,1.49618,1.49618 --c2 1.49618,1.49618,1.49618,1.49618,1.49618 -s 30 -l 8 --max_steps 30 -e 10 -f 0 -g 1 --log_file=log/ippso_cnn_2031.log --gbest_file=log/gbest_2031.pkl &
```

**Search the best particle with velocity clamping**

```bash
nohup python3 main.py -m 1 --w 0.7298 --c1 1.49618,1.49618,1.49618,1.49618,1.49618 --c2 1.49618,1.49618,1.49618,1.49618,1.49618 -v 0.4,25.6,25.6,25.6,25.6 -s 30 -l 8 --max_steps 30 -e 10 -f 0 -g 1 --log_file=log/ippso_cnn_2031.log --gbest_file=log/gbest_2031.pkl &
```

**Train the best particle**

```bash
nohup python3 main.py -m 1 -e 100 -f 0 -g 1 -o 1 --log_file=log/ippso_cnn_optimise_2031.log --gbest_file=log/gbest_2031.pkl &
```

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
nohup python3 main.py -d mdrbi -m 1 --w 0.7298 --c1 1.49618,1.49618,1.49618,1.49618,1.49618 --c2 1.49618,1.49618,1.49618,1.49618,1.49618 -s 30 -l 8 --max_steps 30 -e 10 -r 0.01 -f 0 -g 1 --log_file=log/ippso_cnn_306.log --gbest_file=log/gbest_306.pkl &
```

**Search the best particle with velocity clamping**

```bash
nohup python3 main.py -d mdrbi -m 1 --w 0.7298 --c1 1.49618,1.49618,1.49618,1.49618,1.49618 --c2 1.49618,1.49618,1.49618,1.49618,1.49618 -v 0.4,25.6,25.6,25.6,25.6 -s 30 -l 8 --max_steps 30 -e 10 -r 0.01 -f 0 -g 1 --log_file=log/ippso_cnn_306.log --gbest_file=log/gbest_306.pkl &
```

**Train the best particle**

```bash
nohup python3 main.py -d mdrbi -m 1 -e 100 -r 0.01 --dropout 0.5 -f 0 -g 1 -o 1 --log_file=log/ippso_cnn_optimise_306.log --gbest_file=log/gbest_306.pkl &
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
nohup python3 main.py -d convex -c 2 -m 1 -e 100 -r 0.5 --dropout 0.5 -f 0 -g 1 -o 1 --log_file=log/ippso_cnn_optimise_406.log --gbest_file=log/gbest_406.pkl &
```