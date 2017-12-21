import logging
import argparse
from ippso.pso.population import initialise_cnn_population
from ippso.pso.evaluator import initialise_cnn_evaluator
from ippso.pso.particle import save_particle

def main(args):
    logging.basicConfig(filename='log/ippso_cnn.log', level=logging.DEBUG)
    logging.info('===Load data: %s===', args.dataset)
    loaded_data = _load_data(args.dataset)
    logging.info('===Data loaded===')
    logging.info('===Started===')
    if loaded_data is not None:
        evaluator = initialise_cnn_evaluator(training_data=loaded_data.train['images'], training_label=loaded_data.train['labels'],
                                         validation_data=loaded_data.test['images'], validation_label=loaded_data.test['labels'])
    else:
        evaluator = None
    pso_pop = initialise_cnn_population(evaluator=evaluator)
    best_particle = pso_pop.fly_2_end()
    save_particle(best_particle)
    logging.info('===Finished===')

def _load_data(dataset_name):
    loaded_data = None
    if dataset_name == 'mb':
        from ippso.data.mb import loaded_data
    elif dataset_name == 'mdrbi':
        from ippso.data.mdrbi import loaded_data
    elif dataset_name == 'convex':
        from ippso.data.convex import loaded_data
    return loaded_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help='choose a dataset among mb, mdrbi, or convex')
    args = parser.parse_args()
    main(args)
