import bessel
from PySpice.Unit import u_A
import time
import numpy

TEST_NAME = "op_point_current"

if __name__ == "__main__":
    if not os.path.exists('./results/'):
        os.mkdirs('./results/')
    pC = bessel.plant_circuit("Detector",
                              lambda : 2.1@u_A,
                              bessel.Signal.CURRENT)
    pC.characterize_plant(-6.00@u_A, 6.00@u_A, 0.01@u_A)
    pC.plot_th_tc(bessel.IndVar.CURRENT)
    plt.savefig('./figs/{}'.format(TEST_NAME))
    data = numpy.array([pC.get_i_arr(), pC.get_th_actual()])
    numpy.save("./results/{}_curr_th_characterization".format(TEST_NAME), data)
    data = numpy.array([pC.get_i_arr(), pC.get_tc_actual()])
    numpy.save("./results/{}_curr_tc_characterization".format(TEST_NAME), data)
    plt.show()
