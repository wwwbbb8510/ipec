#!/usr/bin/env bash
EPOCH=3
for i in `seq 1 ${EPOCH}`;do
    python3 main.py -d mbi -m 1 --partial_dataset 0.1 --ip_structure 2 --re_evolve 1 --w 0.7298 --c1 1.49618,1.49618 --c2 1.49618,1.49618 -v 4,25.6 -s 30 -l 8 --max_steps 10 -e 10 -f 0 -g 1 --log_file=log/ippso_cnn_mbi_1003.log --gbest_file=log/gbest_mbi_1003.pkl
done