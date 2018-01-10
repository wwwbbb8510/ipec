#!/usr/bin/env bash
while getopts g:i:e: option;do
    case "${option}" in
    g) GPU=${OPTARG};;
    i) ID=${OPTARG};;
    e) EVOLVE=${OPTARG};;
    esac
done

print_help(){
    printf "Parameter g(GPU) and i(ID) are mandatory\n"
    printf "g values - 1:first gpu, 2: second gpu"
    printf "e values - 1:PSO evolve particles, 2: training the final model"
    exit 1
}

if [ -z "${GPU}" -o -z "${ID}" ];then
    print_help
fi

FIRST_GPU_ID=0
MAX_GPU=1

if [ $EVOLVE -eq 1 ];then
    case "${GPU}" in
        1)
        FIRST_GPU_ID=0
        printf "evolve CNN for mbi, mrb and mrd"
        python3 main.py -d mbi -m 1 --partial_dataset 0.1 --ip_structure 2 --w 0.7298 --c1 1.49618,1.49618 --c2 1.49618,1.49618 -v 4,25.6 -s 30 -l 8 --max_steps 30 -e 10 -f ${FIRST_GPU_ID} -g 1 --log_file=log/ippso_cnn_mbi_${ID}.log --gbest_file=log/gbest_mbi_${ID}.pkl
        python3 main.py -d mrb -m 1 --partial_dataset 0.1 --ip_structure 2 --w 0.7298 --c1 1.49618,1.49618 --c2 1.49618,1.49618 -v 4,25.6 -s 30 -l 8 --max_steps 30 -e 10 -f ${FIRST_GPU_ID} -g 1 --log_file=log/ippso_cnn_mrb_${ID}.log --gbest_file=log/gbest_mrb_${ID}.pkl
        python3 main.py -d mrd -m 1 --partial_dataset 0.1 --ip_structure 2 --w 0.7298 --c1 1.49618,1.49618 --c2 1.49618,1.49618 -v 4,25.6 -s 30 -l 8 --max_steps 30 -e 10 -f ${FIRST_GPU_ID} -g 1 --log_file=log/ippso_cnn_mrd_${ID}.log --gbest_file=log/gbest_mrd__${ID}.pkl
        ;;
        2)
        FIRST_GPU_ID=1
        printf "evolve CNN for mb, mdrbi and convex"
        python3 main.py -d mb -m 1 --partial_dataset 0.1 --ip_structure 2 --w 0.7298 --c1 1.49618,1.49618 --c2 1.49618,1.49618 -v 4,25.6 -s 30 -l 8 --max_steps 30 -e 10 -f ${FIRST_GPU_ID} -g 1 --log_file=log/ippso_cnn_mb_${ID}.log --gbest_file=log/gbest_mb_${ID}.pkl
        python3 main.py -d mdrbi -m 1 --partial_dataset 0.1 --ip_structure 2 --w 0.7298 --c1 1.49618,1.49618 --c2 1.49618,1.49618 -v 4,25.6 -s 30 -l 8 --max_steps 30 -e 10 -f ${FIRST_GPU_ID} -g 1 --log_file=log/ippso_cnn_mdrbi_${ID}.log --gbest_file=log/gbest_mdrbi_${ID}.pkl
        python3 main.py -d convex -m 1 --partial_dataset 0.1 --ip_structure 2 --w 0.7298 --c1 1.49618,1.49618 --c2 1.49618,1.49618 -v 4,25.6 -s 30 -l 8 --max_steps 30 -e 10 -f ${FIRST_GPU_ID} -g 1 --log_file=log/ippso_cnn_convex_${ID}.log --gbest_file=log/gbest_convex_${ID}.pkl
        ;;
        *)
        print_help
        ;;
    esac
elif [ $EVOLVE -eq 1 ];then
    case "${GPU}" in
        1)
        FIRST_GPU_ID=0
        printf "train CNN for mbi, mrb and mrd"
        python3 main.py -d mbi -m 1 --ip_structure 2 -e 100 -f ${FIRST_GPU_ID} -g 1 -o 1 --log_file=log/ippso_cnn_optimise_mbi_${ID}.log --gbest_file=log/gbest_mbi_${ID}.pkl
        python3 main.py -d mrb -m 1 --ip_structure 2 -e 100 -f ${FIRST_GPU_ID} -g 1 -o 1 --log_file=log/ippso_cnn_optimise_mrb_${ID}.log --gbest_file=log/gbest_mrb_${ID}.pkl
        python3 main.py -d mrd -m 1 --ip_structure 2 -e 100 -f ${FIRST_GPU_ID} -g 1 -o 1 --log_file=log/ippso_cnn_optimise_mrd_${ID}.log --gbest_file=log/gbest_mrd_${ID}.pkl
        ;;
        2)
        FIRST_GPU_ID=1
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

