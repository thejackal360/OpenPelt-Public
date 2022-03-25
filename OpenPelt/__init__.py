from .OpenPelt import tec_plant, rc_ckt_plant, op_amp_plant, sequencer, circular_buffer_sequencer, TECPlate, Signal, IndVar
from .controller import controller, random_agent_controller, bang_bang_controller, pid_controller

try:
    from .controller import fake_neural_controller
    INCLUDE_TORCH = True
except ImportError:
    INCLUDE_TORCH = False
    print("Warning: cannot import torch")
