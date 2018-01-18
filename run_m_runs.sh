#!/usr/bin/env bash
while getopts r:p:e:f: option;do
    case "${option}" in
    r) RUNS=${OPTARG};;
    p) PROGRAM_ID=${OPTARG};;
    e) EVOLVE=${OPTARG};;
    f) FIRST_GPU_ID=${OPTARG};;
    esac
done

print_help(){
    printf "Parameter r(RUNS) is mandatory\n"
    printf "r values - number of runs"
    printf "p values - program ID"
    printf "e values - 1:PSO evolve particles, 2: training the final model"
    printf "f values - first gpu id"
    exit 1
}

if [ -z "${RUNS}" -o -z "${PROGRAM_ID}" -o -z "${EVOLVE}" -o -z "${FIRST_GPU_ID}" ];then
    print_help
fi

for i in `seq 1 ${RUNS}`;do
    bash run_all.sh -g 2 -i ${PROGRAM_ID}00${i} -e ${EVOLVE} -f ${FIRST_GPU_ID}
    sleep 120
done