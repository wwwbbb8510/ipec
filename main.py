import logging
import argparse
import numpy as np
from ippso.pso.population import initialise_cnn_population
from ippso.pso.evaluator import initialise_cnn_evaluator
from ippso.pso.particle import save_particle, load_particle
from ippso.cnn.layers import ConvLayer
from ippso.cnn.layers import PoolingLayer
from ippso.cnn.layers import FullyConnectedLayer
from ippso.cnn.layers import DisabledLayer
from ippso.cnn.layers import initialise_cnn_layers_3_bytes, initialise_cnn_layers_with_xavier_weights


def main(args):
    _filter_args(args)
    if args.optimise == 1:
        _optimise_learned_particle(args)
    else:
        _pso_search(args)


def _optimise_learned_particle(args):
    """
    optimise the learned particle

    :param args: arguments
    """
    if args.log_file is None:
        logging.basicConfig(filename='log/ippso_cnn_optimise.log', level=logging.DEBUG)
    else:
        logging.basicConfig(filename=args.log_file, level=logging.DEBUG)
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
                                             optimise=True)
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
                                             optimise=True)
    loaded_particle = load_particle(args.gbest_file)
    evaluator.eval(loaded_particle)
    logging.info('===Finished===')


def _pso_search(args):
    """
    use IPPSO to search the best particle

    :param args: arguments
    """
    if args.log_file is None:
        logging.basicConfig(filename='log/ippso_cnn.log', level=logging.DEBUG)
    else:
        logging.basicConfig(filename=args.log_file, level=logging.DEBUG)
    logging.info('===Load data - dataset:%s, mode:%s===', args.dataset, args.mode)
    loaded_data = _load_data(args.dataset, args.mode, args.partical_dataset)
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
    pso_pop = initialise_cnn_population(pop_size=args.pop_size, particle_length=args.particle_length,
                                        evaluator=evaluator, w=args.w, c1=args.c1, c2=args.c2,
                                        max_fully_connected_length=args.max_full,
                                        layers=layers, v_max=args.v_max)
    best_particle = pso_pop.fly_2_end(max_steps=args.max_steps)
    save_particle(best_particle, args.gbest_file)
    logging.info('===Finished===')


def _load_data(dataset_name, mode, partical_dataset=None):
    """
    load the dataset

    :param dataset_name: dataset name
    :param mode: mode
    :return: loaded data
    """
    loaded_data = None
    from ippso.data.core import DataLoader
    DataLoader.mode = mode
    DataLoader.partical_dataset = partical_dataset
    if dataset_name == 'mb':
        from ippso.data.mb import loaded_data
    elif dataset_name == 'mdrbi':
        from ippso.data.mdrbi import loaded_data
    elif dataset_name == 'convex':
        from ippso.data.convex import loaded_data
    return loaded_data


def _filter_args(args):
    """
    filter the arguments

    :param args: arguments
    """
    args.class_num = int(args.class_num) if args.class_num is not None else None
    args.pop_size = int(args.pop_size) if args.pop_size is not None else None
    args.particle_length = int(args.particle_length) if args.particle_length is not None else None
    args.max_full = int(args.max_full) if args.max_full is not None else None
    args.max_steps = int(args.max_steps) if args.max_steps is not None else None
    args.training_epoch = int(args.training_epoch) if args.training_epoch is not None else None
    args.first_gpu_id = int(args.first_gpu_id) if args.first_gpu_id is not None else None
    args.max_gpu = int(args.max_gpu) if args.max_gpu is not None else None
    args.optimise = int(args.optimise) if args.optimise is not None else 0
    args.w = float(args.w) if args.w is not None else None
    args.c1 = np.asarray(args.c1.split(',')).astype(np.float) if args.c1 is not None else None
    args.c2 = np.asarray(args.c2.split(',')).astype(np.float) if args.c2 is not None else None
    args.v_max = np.asarray(args.v_max.split(',')).astype(np.float) if args.v_max is not None else None
    args.regularise = float(args.regularise) if args.regularise is not None else 0
    args.dropout = float(args.dropout) if args.dropout is not None else 0
    args.ip_structure = int(args.ip_structure) if args.ip_structure is not None else 0
    args.partical_dataset = float(args.partical_dataset) if args.partical_dataset is not None else None

# main entrance
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help='choose a dataset among mb, mdrbi, or convex')
    parser.add_argument('-c', '--class_num', help='# of classes for the classification problem')
    parser.add_argument('-m', '--mode', help='default:None, 1: production (load full data)')
    parser.add_argument('-s', '--pop_size', help='population size')
    parser.add_argument('-l', '--particle_length', help='particle max length')
    parser.add_argument('--max_full', help='max fully connected layers')
    parser.add_argument('--max_steps', help='max fly steps')
    parser.add_argument('-e', '--training_epoch', help='training epoch for the evaluation')
    parser.add_argument('-f', '--first_gpu_id', help='first gpu id')
    parser.add_argument('-g', '--max_gpu', help='max number of gpu')
    parser.add_argument('-o', '--optimise',
                        help='optimise the learned CNN architecture. Default: None. 1: optimise; otherwise IPPSO search')
    parser.add_argument('--log_file', help='the path of log file')
    parser.add_argument('--gbest_file', help='the path of gbest file')
    parser.add_argument('--w', help='w parameter of PSO')
    parser.add_argument('--c1', help='c1 parameter of PSO')
    parser.add_argument('--c2', help='c2 parameter of PSO')
    parser.add_argument('-v', '--v_max', help='PSO max velocity used by velocity clamping')
    parser.add_argument('-r', '--regularise',  help='weight regularisation hyper-parameter. ')
    parser.add_argument('--dropout', help='enable dropout and set dropout rate')
    parser.add_argument('--ip_structure',
                        help='IP structure. default: 5 bytes, 1: 3 bytes, 2: 2 bytes with xavier weight initialisation')
    parser.add_argument('--partical_dataset',
                        help='Use partical dataset for learning CNN architecture to speed up the learning process.')

    args = parser.parse_args()
    main(args)
