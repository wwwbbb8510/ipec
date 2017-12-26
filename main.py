import logging
import argparse
from ippso.pso.population import initialise_cnn_population
from ippso.pso.evaluator import initialise_cnn_evaluator
from ippso.pso.particle import save_particle, load_particle


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
    logging.basicConfig(filename='log/ippso_cnn_optimise.log', level=logging.DEBUG)
    logging.info('===Load data - dataset:%s, mode:%s===', args.dataset, args.mode)
    loaded_data = _load_data(args.dataset, args.mode)
    logging.info('===Data loaded===')
    logging.info('===Started===')
    if loaded_data is not None:
        evaluator = initialise_cnn_evaluator(training_epoch=args.training_epoch,
                                             training_data=loaded_data.train['images'],
                                             training_label=loaded_data.train['labels'],
                                             validation_data=loaded_data.test['images'],
                                             validation_label=loaded_data.test['labels'], max_gpu=args.max_gpu)
    else:
        evaluator = initialise_cnn_evaluator(training_epoch=args.training_epoch, max_gpu=args.max_gpu)
    loaded_particle = load_particle()
    evaluator.eval(loaded_particle)
    logging.info('===Finished===')


def _pso_search(args):
    """
    use IPPSO to search the best particle

    :param args: arguments
    """
    logging.basicConfig(filename='log/ippso_cnn.log', level=logging.DEBUG)
    logging.info('===Load data - dataset:%s, mode:%s===', args.dataset, args.mode)
    loaded_data = _load_data(args.dataset, args.mode)
    logging.info('===Data loaded===')
    logging.info('===Started===')
    if loaded_data is not None:
        evaluator = initialise_cnn_evaluator(training_epoch=args.training_epoch,
                                             training_data=loaded_data.train['images'],
                                             training_label=loaded_data.train['labels'],
                                             validation_data=loaded_data.test['images'],
                                             validation_label=loaded_data.test['labels'], max_gpu=args.max_gpu)
    else:
        evaluator = initialise_cnn_evaluator(training_epoch=args.training_epoch, max_gpu=args.max_gpu)
    pso_pop = initialise_cnn_population(pop_size=args.pop_size, particle_length=args.particle_length,
                                        evaluator=evaluator)
    best_particle = pso_pop.fly_2_end(max_steps=args.max_steps)
    save_particle(best_particle)
    logging.info('===Finished===')


def _load_data(dataset_name, mode):
    """
    load the dataset

    :param dataset_name: dataset name
    :param mode: mode
    :return: loaded data
    """
    loaded_data = None
    from ippso.data.core import DataLoader
    DataLoader.mode = mode
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
    args.pop_size = int(args.pop_size) if args.pop_size is not None else None
    args.particle_length = int(args.particle_length) if args.particle_length is not None else None
    args.max_steps = int(args.max_steps) if args.max_steps is not None else None
    args.training_epoch = int(args.training_epoch) if args.training_epoch is not None else None
    args.max_gpu = int(args.max_gpu) if args.max_gpu is not None else None
    args.optimise = int(args.optimise) if args.optimise is not None else 0

# main entrance
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help='choose a dataset among mb, mdrbi, or convex')
    parser.add_argument('-m', '--mode', help='default:None, 1: production (load full data)')
    parser.add_argument('-s', '--pop_size', help='population size')
    parser.add_argument('-l', '--particle_length', help='particle max length')
    parser.add_argument('--max_steps', help='max fly steps')
    parser.add_argument('-e', '--training_epoch', help='training epoch for the evaluation')
    parser.add_argument('-g', '--max_gpu', help='max number of gpu')
    parser.add_argument('-o', '--optimise', help='optimise the learned CNN architecture. Default: None. 1: optimise; otherwise IPPSO search')

    args = parser.parse_args()
    main(args)
