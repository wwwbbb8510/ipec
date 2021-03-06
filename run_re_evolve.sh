#!/usr/bin/env bash
while getopts g:i:d: option;do
    case "${option}" in
    g) GPU=${OPTARG};;
    i) ID=${OPTARG};;
    d) DATASET=${OPTARG};;
    esac
done

print_help(){
    printf "Parameter g(GPU), i(ID), d(dataset) are mandatory\n"
    printf "g values - 0:first gpu, 1: second gpu"
    printf "e values - 1:PSO evolve particles, 2: training the final model"
    printf "d values - choose a dataset among mb, mbi, mdrbi, mrb, mrd or convex"
    exit 1
}

if [ -z "${GPU}" -o -z "${ID}" -o -z "${DATASET}" ];then
    print_help
fi

EPOCH=10
for i in `seq 1 ${EPOCH}`;do
    python3 main.py -d ${DATASET} -m 1 --partial_dataset 0.1 --ip_structure 2 --re_evolve 1 --w 0.7298 --c1 1.49618,1.49618 --c2 1.49618,1.49618 -v 4,25.6 -s 30 -l 8 --max_steps 10 -e 10 -f ${GPU} -g 1 --log_file=log/ippso_cnn_${DATASET}_${ID}.log --gbest_file=log/gbest_${DATASET}_${ID}.pkl
done