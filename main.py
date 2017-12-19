import logging
from ippso.pso.population import initialise_cnn_population
from ippso.pso.particle import save_particle

def main():
    logging.basicConfig(filename='log/ippso_cnn.log', level=logging.DEBUG)
    logging.info('===Started===')
    pso_pop = initialise_cnn_population()
    best_particle = pso_pop.fly_2_end()
    save_particle(best_particle)
    logging.info('===Finished===')


if __name__ == '__main__':
    main()

