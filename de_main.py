import argparse
import logging
import os

import numpy as np

from ipec.cnn.evaluator import initialise_cnn_evaluator
from ipec.cnn.layers import initialise_cnn_layers_3_bytes, initialise_cnn_layers_with_xavier_weights
from ipec.de.agent import save_agent, load_agent
from ipec.de.population import initialise_cnn_population


def main(args):
    _filter_args(args)
    if args.optimise == 1:
        _optimise_learned_agent(args)
    else:
        _de_search(args)


def _optimise_learned_agent(args):
    """
    optimise the learned agent

    :param args: arguments
    """
    if args.log_file is None:
        log_file_path = 'log/ipde_cnn_optimise.log'
    else:
        log_file_path = args.log_file
    tensorboard_path = None
    if args.use_tensorboard == 1:
        tensorboard_path = os.path.join(os.path.splitext(log_file_path)[0], 'tensorboard')
        if not os.path.exists(tensorboard_path):
            os.makedirs(tensorboard_path)
    logging.basicConfig(filename=log_file_path, level=logging.DEBUG)
    logging.info('===Load data - dataset:%s, mode:%s===', args.dataset, args.mode)
    loaded_data = _load_data(args.dataset, args.mode)
    logging.info('===Data loaded===')
    logging.info('===Started===')
    if loaded_data is not None:
        evaluator = initialise_cnn_evaluator(training_epoch=args.training_epoch,
                                             training_data=loaded_data.train['images'],
                                             training_label=loaded_data.train['labels'],
                                             validation_data=loaded_data.validation['images'],
                                             validation_label=loaded_data.validation['labels'],
                                             max_gpu=args.max_gpu,
                                             first_gpu_id=args.first_gpu_id,
                                             class_num=args.class_num,
                                             regularise=args.regularise,
                                             dropout=args.dropout,
                                             mean_centre=7,
                                             mean_divisor=80,
                                             stddev_divisor=16,
                                             test_data=loaded_data.test['images'],
                                             test_label=loaded_data.test['labels'],
                                             optimise=True,
                                             tensorboard_path=tensorboard_path)
    else:
        evaluator = initialise_cnn_evaluator(training_epoch=args.training_epoch,
                                             max_gpu=args.max_gpu,
                                             first_gpu_id=args.first_gpu_id,
                                             class_num=args.class_num,
                                             regularise=args.regularise,
                                             dropout=args.dropout,
                                             mean_centre=7,
                                             mean_divisor=80,
                                             stddev_divisor=16,
                                             optimise=True,
                                             tensorboard_path=tensorboard_path)
    loaded_agent = load_agent(args.gbest_file)
    evaluator.eval(loaded_agent)
    logging.info('===Finished===')


def _de_search(args):
    """
    use IPDE to search the best agent

    :param args: arguments
    """
    if args.log_file is None:
        logging.basicConfig(filename='log/ipde_cnn.log', level=logging.DEBUG)
    else:
        logging.basicConfig(filename=args.log_file, level=logging.DEBUG)
    logging.info('===Load data - dataset:%s, mode:%s===', args.dataset, args.mode)
    loaded_data = _load_data(args.dataset, args.mode, args.partial_dataset)
    logging.info('===Data loaded===')
    logging.info('===Started===')
    if loaded_data is not None:
        evaluator = initialise_cnn_evaluator(training_epoch=args.training_epoch,
                                             training_data=loaded_data.train['images'],
                                             training_label=loaded_data.train['labels'],
                                             validation_data=loaded_data.validation['images'],
                                             validation_label=loaded_data.validation['labels'],
                                             max_gpu=args.max_gpu,
                                             first_gpu_id=args.first_gpu_id,
                                             class_num=args.class_num,
                                             regularise=args.regularise,
                                             dropout=args.dropout,
                                             mean_centre=7,
                                             mean_divisor=80,
                                             stddev_divisor=16,
                                             test_data=loaded_data.test['images'],
                                             test_label=loaded_data.test['labels'])
    else:
        evaluator = initialise_cnn_evaluator(training_epoch=args.training_epoch,
                                             max_gpu=args.max_gpu,
                                             first_gpu_id=args.first_gpu_id,
                                             class_num=args.class_num,
                                             regularise=args.regularise,
                                             dropout=args.dropout,
                                             mean_centre=7,
                                             mean_divisor=80,
                                             stddev_divisor=16
                                             )
    if args.ip_structure == 1:
        layers = initialise_cnn_layers_3_bytes()
    elif args.ip_structure == 2:
        layers = initialise_cnn_layers_with_xavier_weights()
    else:
        layers = None
    de_pop = initialise_cnn_population(pop_size=args.pop_size, agent_length=args.agent_length,
                                        evaluator=evaluator, f=args.f_weight, cr=args.cr,
                                        max_fully_connected_length=args.max_full,
                                        layers=layers, max_generation=args.max_generation)
    best_agent = de_pop.evolve()
    save_agent(best_agent, args.gbest_file)
    logging.info('===Finished===')


def _load_data(dataset_name, mode, partial_dataset=None):
    """
    load the dataset

    :param dataset_name: dataset name
    :param mode: mode
    :return: loaded data
    """
    loaded_data = None
    from ipec.data.core import DataLoader
    DataLoader.mode = mode
    DataLoader.partial_dataset = partial_dataset
    if dataset_name == 'mb':
        from ipec.data.mb import loaded_data
    elif dataset_name == 'mbi':
        from ipec.data.mbi import loaded_data
    elif dataset_name == 'mdrbi':
        from ipec.data.mdrbi import loaded_data
    elif dataset_name == 'mrb':
        from ipec.data.mrb import loaded_data
    elif dataset_name == 'mrd':
        from ipec.data.mrd import loaded_data
    elif dataset_name == 'convex':
        from ipec.data.convex import loaded_data
    return loaded_data


def _filter_args(args):
    """
    filter the arguments

    :param args: arguments
    """
    args.class_num = int(args.class_num) if args.class_num is not None else None
    args.pop_size = int(args.pop_size) if args.pop_size is not None else None
    args.agent_length = int(args.agent_length) if args.agent_length is not None else None
    args.max_full = int(args.max_full) if args.max_full is not None else None
    args.max_generation = int(args.max_generation) if args.max_generation is not None else None
    args.training_epoch = int(args.training_epoch) if args.training_epoch is not None else None
    args.first_gpu_id = int(args.first_gpu_id) if args.first_gpu_id is not None else None
    args.max_gpu = int(args.max_gpu) if args.max_gpu is not None else None
    args.optimise = int(args.optimise) if args.optimise is not None else 0
    args.f_weight = float(args.f_weight) if args.f_weight is not None else None
    args.cr = float(args.cr) if args.cr is not None else None
    args.regularise = float(args.regularise) if args.regularise is not None else 0
    args.dropout = float(args.dropout) if args.dropout is not None else 0
    args.ip_structure = int(args.ip_structure) if args.ip_structure is not None else 0
    args.partial_dataset = float(args.partial_dataset) if args.partial_dataset is not None else None
    args.use_tensorboard = int(args.use_tensorboard) if args.use_tensorboard is not None else 0

# main entrance
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help='choose a dataset among mb, mbi, mdrbi, mrb, mrd or convex')
    parser.add_argument('-c', '--class_num', help='# of classes for the classification problem')
    parser.add_argument('-m', '--mode', help='default:None, 1: production (load full data)')
    parser.add_argument('-s', '--pop_size', help='population size')
    parser.add_argument('-l', '--agent_length', help='agent max length')
    parser.add_argument('--max_full', help='max fully connected layers')
    parser.add_argument('--max_generation', help='max fly steps')
    parser.add_argument('-e', '--training_epoch', help='training epoch for the evaluation')
    parser.add_argument('-f', '--first_gpu_id', help='first gpu id')
    parser.add_argument('-g', '--max_gpu', help='max number of gpu')
    parser.add_argument('-o', '--optimise',
                        help='optimise the learned CNN architecture. Default: None. 1: optimise; otherwise IPDE search')
    parser.add_argument('--log_file', help='the path of log file')
    parser.add_argument('--gbest_file', help='the path of gbest file')
    parser.add_argument('--f_weight', help='differential weight')
    parser.add_argument('--cr', help='crossover rate')
    parser.add_argument('-r', '--regularise',  help='weight regularisation hyper-parameter. ')
    parser.add_argument('--dropout', help='enable dropout and set dropout rate')
    parser.add_argument('--ip_structure',
                        help='IP structure. default: 5 bytes, 1: 3 bytes, 2: 2 bytes with xavier weight initialisation')
    parser.add_argument('--partial_dataset',
                        help='Use partial dataset for learning CNN architecture to speed up the learning process.')
    parser.add_argument('--use_tensorboard',
                        help='indicate whether to use tensorboard. default: not use, 1: use')

    args = parser.parse_args()
    main(args)
