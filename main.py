from ippso.pso.population import initialise_cnn_population


if __name__ == '__main__':
    pso_pop = initialise_cnn_population()
    pso_pop.fly_2_end()