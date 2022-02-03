import bessel
from PySpice.Unit import u_A
import time
import numpy

TEST_NAME = "transient"

if __name__ == "__main__":
    if not os.path.exists('./results/'):
        os.mkdirs('./results/')
    pC = bessel.plant_circuit("Detector",
                              lambda : 2.1@u_A,
                              bessel.Signal.CURRENT)
    start_t = time.time()
    pC.run_sim()
    end_t = time.time()
    print("Run Time : {} [s] , Sim Time : {} [s] , Speedup Factor : {}".format(
           end_t - start_t, SIMULATION_TIME_IN_SEC,
           SIMULATION_TIME_IN_SEC / (end_t - start_t)))
    pC.plot_th_tc(bessel.IndVar.TIME)
    plt.savefig('./figs/{}'.format(TEST_NAME))
    data = numpy.array([pC.get_t(), pC.get_th_sensor()])
    numpy.save('./results/{}_time_th_sensor_curr'.format(TEST_NAME), data)
    data = numpy.array([pC.get_t(), pC.get_tc_sensor()])
    numpy.save('./results/{}_time_tc_sensor_curr'.format(TEST_NAME), data)
    data = numpy.array([pC.get_t(), pC.get_i_arr()])
    numpy.save('./results/{}_time_i_curr'.format(TEST_NAME), data)
    plt.show()
