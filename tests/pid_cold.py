import bessel
import time
import numpy

TEST_NAME = "pid_cold"

if __name__ == "__main__":
    if not os.path.exists('./results/'):
        os.mkdirs('./results/')
    pC = bessel.plant_circuit("Detector", None, bessel.Signal.VOLTAGE)
    cbs = bessel.circular_buffer_sequencer([50.00, 30.00, 40.00], pC.get_ncs(), bessel.TECPlate.COLD_SIDE)
    pidc = bessel.pid_controller(cbs, 8.00, 0.00, 0.00)
    pC.set_controller_f(bbc.controller_f)
    pC.run_sim()
    pC.plot_th_tc(bessel.IndVar.TIME, plot_driver = False)
    plt.savefig('./figs/{}'.format(TEST_NAME))
    data = numpy.array([pC.get_t(), pC.get_th_sensor()])
    numpy.save('./results/{}_time_th_sensor_curr'.format(TEST_NAME), data)
    data = numpy.array([pC.get_t(), pC.get_tc_sensor()])
    numpy.save('./results/{}_time_tc_sensor_curr'.format(TEST_NAME), data)
    data = numpy.array([pC.get_t(), pC.get_v_arr()])
    numpy.save('./results/{}_time_v_curr'.format(TEST_NAME), data)
    plt.show()
