#!/usr/bin/env bash
while getopts g:i:e:f: option;do
    case "${option}" in
    g) GPU=${OPTARG};;
    i) ID=${OPTARG};;
    e) EVOLVE=${OPTARG};;
    f) FIRST_GPU_ID=${OPTARG};;
    esac
done

print_help(){
    printf "Parameter g(GPU), i(ID), e(Evolve or train) are mandatory\n"
    printf "g values - group. 1: mbi, mrb, mrd 2: mb, mdrbi, convex"
    printf "e values - 1:PSO evolve particles, 2: training the final model"
    printf "f values - first gpu id"
    exit 1
}

if [ -z "${GPU}" -o -z "${ID}" -o -z "${EVOLVE}" ];then
    print_help
fi

MAX_GPU=1
PARTIAL_10=" --partial_dataset 0.1 "
PARTIAL_15=" --partial_dataset 0.15 "

# disable using partial
#PARTIAL_10=""
#PARTIAL_15=""

if [ "${EVOLVE}" -eq "1" ];then
    case "${GPU}" in
        1)
        if [ -z "${FIRST_GPU_ID}" ];then
            FIRST_GPU_ID=0
        fi
        printf "evolve CNN for mbi, mrb and mrd"
        python3 main.py -d mbi -m 1 ${PARTIAL_10} --ip_structure 2 --w 0.7298 --c1 1.49618,1.49618 --c2 1.49618,1.49618 -v 4,25.6 -s 30 -l 8 --max_steps 30 -e 10 -f ${FIRST_GPU_ID} -g 1 --log_file=log/ippso_cnn_mbi_${ID}.log --gbest_file=log/gbest_mbi_${ID}.pkl
        python3 main.py -d mrb -m 1 ${PARTIAL_10} --ip_structure 2 --w 0.7298 --c1 1.49618,1.49618 --c2 1.49618,1.49618 -v 4,25.6 -s 30 -l 8 --max_steps 30 -e 10 -f ${FIRST_GPU_ID} -g 1 --log_file=log/ippso_cnn_mrb_${ID}.log --gbest_file=log/gbest_mrb_${ID}.pkl
        python3 main.py -d mrd -m 1 ${PARTIAL_10} --ip_structure 2 --w 0.7298 --c1 1.49618,1.49618 --c2 1.49618,1.49618 -v 4,25.6 -s 30 -l 8 --max_steps 30 -e 10 -f ${FIRST_GPU_ID} -g 1 --log_file=log/ippso_cnn_mrd_${ID}.log --gbest_file=log/gbest_mrd_${ID}.pkl
        ;;
        2)
        if [ -z "${FIRST_GPU_ID}" ];then
            FIRST_GPU_ID=1
        fi
        printf "evolve CNN for mb, mdrbi and convex"
        python3 main.py -d mb -m 1 ${PARTIAL_10} --ip_structure 2 --w 0.7298 --c1 1.49618,1.49618 --c2 1.49618,1.49618 -v 4,25.6 -s 30 -l 8 --max_steps 30 -e 10 -f ${FIRST_GPU_ID} -g 1 --log_file=log/ippso_cnn_mb_${ID}.log --gbest_file=log/gbest_mb_${ID}.pkl
        python3 main.py -d mdrbi -m 1 ${PARTIAL_10} --ip_structure 2 --w 0.7298 --c1 1.49618,1.49618 --c2 1.49618,1.49618 -v 4,25.6 -s 30 -l 8 --max_steps 30 -e 10 -f ${FIRST_GPU_ID} -g 1 --log_file=log/ippso_cnn_mdrbi_${ID}.log --gbest_file=log/gbest_mdrbi_${ID}.pkl
        python3 main.py -d convex -m 1 ${PARTIAL_15} --ip_structure 2 --w 0.7298 --c1 1.49618,1.49618 --c2 1.49618,1.49618 -v 4,25.6 -s 30 -l 8 --max_steps 30 -e 10 -f ${FIRST_GPU_ID} -g 1 --log_file=log/ippso_cnn_convex_${ID}.log --gbest_file=log/gbest_convex_${ID}.pkl
        ;;
        *)
        print_help
        ;;
    esac
elif [ "${EVOLVE}" -eq "2" ];then
    case "${GPU}" in
        1)
        if [ -z "${FIRST_GPU_ID}" ];then
            FIRST_GPU_ID=0
        fi
        printf "train CNN for mbi, mrb and mrd"
        python3 main.py -d mbi -m 1 --ip_structure 2 -e 100 -f ${FIRST_GPU_ID} -g 1 -o 1 --log_file=log/ippso_cnn_optimise_mbi_${ID}.log --gbest_file=log/gbest_mbi_${ID}.pkl
        python3 main.py -d mrb -m 1 --ip_structure 2 -e 100 -f ${FIRST_GPU_ID} -g 1 -o 1 --log_file=log/ippso_cnn_optimise_mrb_${ID}.log --gbest_file=log/gbest_mrb_${ID}.pkl
        python3 main.py -d mrd -m 1 --ip_structure 2 -e 100 -f ${FIRST_GPU_ID} -g 1 -o 1 --log_file=log/ippso_cnn_optimise_mrd_${ID}.log --gbest_file=log/gbest_mrd_${ID}.pkl
        ;;
        2)
        if [ -z "${FIRST_GPU_ID}" ];then
            FIRST_GPU_ID=1
        fi
        printf "train CNN for mb, mdrbi and convex"
        python3 main.py -d mb -m 1 --ip_structure 2 -e 100 -f ${FIRST_GPU_ID} -g 1 -o 1 --log_file=log/ippso_cnn_optimise_mb_${ID}.log --gbest_file=log/gbest_mb_${ID}.pkl
        python3 main.py -d mdrbi -m 1 --ip_structure 2 -e 100 -f ${FIRST_GPU_ID} -g 1 -o 1 --log_file=log/ippso_cnn_optimise_mdrbi_${ID}.log --gbest_file=log/gbest_mdrbi_${ID}.pkl
        python3 main.py -d convex -m 1 --ip_structure 2 -e 100 -f ${FIRST_GPU_ID} -g 1 -o 1 --log_file=log/ippso_cnn_optimise_convex_${ID}.log --gbest_file=log/gbest_convex_${ID}.pkl
        ;;
        *)
        print_help
        ;;
    esac
else
    print_help
fi

