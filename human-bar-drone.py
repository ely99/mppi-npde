import os
import jax, math
import jax.numpy as jnp
jax.config.update("jax_platform_name", "cpu")

#import matplotlib.pyplot as plt

#npODE and npSDE
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
sess = tf.compat.v1.InteractiveSession()
from npODEeSDE.npde_helper import load_model

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sbmpc import BaseObjective
import sbmpc.settings as settings
from sbmpc.simulation import build_all, Generator_feasible_desired_state, Dataset
from sbmpc.geometry import skew, quat_product, quat2rotm, quat_inverse

os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'

jax.config.update("jax_default_matmul_precision", "high")
script_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(script_dir, "models/huma-bar-drone.xml")  

SCENE_PATH = xml_path
INPUT_MAX = jnp.array([1, 1, 1, 1])
INPUT_MIN = - INPUT_MAX
STARTING_INPUT = jnp.array([0., 0., 0., 0.], dtype=jnp.float32) # this is really the starting input
STARTING_POSITION = jnp.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=jnp.float32) # i don't really know what it is. BUT IT IS NOT THE STARTING POSITION
REFERENCE_POSITION =  Generator_feasible_desired_state(1, 2.5, -3, 60) #half_len of the bar[m], x[m], y[m], theta[deg]
SIM_TIME = 1 # seconds
DT = 0.02
TIME_HORIZON = 30
FILE_NAME = "some_sim"
DIRECTORY = 'SIMULATION_DATA'

class Objective(BaseObjective): # OBJECTIVE/COST FUNCTION

    def __init__(self, reference: jnp.array): #           [Xl,    Yl,     0,  Xr,     Yr,     0, velocities]
        self.Q = jnp.diag(jnp.array([25000., 25000., 0., 25000., 25000., 0., 10., 10., 0., 10., 10., 0.]))
        self.R = jnp.diag(jnp.array([10, 10, 10, 10]))
        self.Qf = jnp.diag(jnp.array([1000., 1000., 0., 1000., 1000., 0., 1000., 1000., 0., 1000., 1000., 0.]))
        self.q_th = 3000 # 1000
        self.r_human = 3000
        self.x_bar = -(reference[0] - reference[3])
        self.y_bar = -(reference[1] - reference[4])
        self.th_bar = math.atan2(self.y_bar, self.x_bar)

    def get_Q(self):
        return self.Q
    
    def get_R(self):
        return self.R
    
    def get_Qf(self):
        return self.Qf
    
    def compute_current_th(self, state: jnp.array):
        delta_y = state[4] - state[1]
        delta_x = state[3]- state[0]
        return jnp.arctan2(delta_y, delta_x)
    
    def compute_state_error(self, state: jnp.array, state_ref : jnp.array) -> jnp.array:
        state_err = state[:] - state_ref[:]
        return state_err
    
    def running_cost(self, state: jnp.array, inputs: jnp.array, reference, pred_input_ode) -> jnp.float32: # type: ignore
        input_human_cost = ((inputs[:2] - pred_input_ode)**2)*self.r_human
        state_err = self.compute_state_error(state, reference)
        th_err = self.th_bar - self.compute_current_th(state)
        cost = state_err.transpose() @ self.Q @ state_err 
        + inputs.transpose() @ self.R @ inputs 
        +(self.q_th**2)*th_err
        +  input_human_cost # inputs error with the npODE prediction
        #+ self.final_cost(state, reference)
        return cost

    def final_cost(self, state, reference):
        state_err = self.compute_state_error(state, reference)
        return state_err.transpose() @ self.Qf @ state_err


if __name__ == "__main__":
    print("Initializing...")
    robot_config = settings.RobotConfig()

    robot_config.robot_scene_path = SCENE_PATH
    robot_config.nq = 6 # position
    robot_config.nv = 6 # velocity
    robot_config.nu = 4 # input
    robot_config.input_min = INPUT_MIN
    robot_config.input_max = INPUT_MAX
    robot_config.q_init = STARTING_POSITION
    
    config = settings.Config(robot_config)

    config.general.visualize = True
    config.MPC.dt = DT # SAMPLING TIME you also need to change the sampling time in the MUJOCO FILE
    config.MPC.horizon = TIME_HORIZON #TIME HORIZON
    config.MPC.std_dev_mppi = jnp.array([5, 5, 5, 5])
    config.MPC.num_parallel_computations = 100
    config.MPC.initial_guess = STARTING_INPUT # STARTING INPUT???
    config.MPC.lambda_mpc = 50.0 # IDK
    config.MPC.smoothing = "Spline" # IDK
    config.MPC.num_control_points = 5 # IDK
    config.MPC.gains = False
    config.sim_iterations = round(SIM_TIME/config.MPC.dt)

    config.solver_dynamics = settings.DynamicsModel.MJX
    config.sim_dynamics = settings.DynamicsModel.MJX

    # x_init = jnp.concatenate([robot_config[settings.ROBOT_Q_INIT_KEY],
    #                  jnp.zeros(robot_config[settings.ROBOT_NV_KEY], dtype=jnp.float32)], axis=0)
    # reference = jnp.concatenate((x_init, INPUT_HOVER))

    #creating the Data variable
    data = Dataset()
    data.set_from_config(config)
    data.set_file_name(FILE_NAME, DIRECTORY)
    data.set_reference(REFERENCE_POSITION)

    print("Loading human model")
    npSde = load_model('npODEeSDE/npde_state_sde.pkl',sess)
    npOde = load_model('npODEeSDE/npde_state.pkl',sess)

    reference = REFERENCE_POSITION
    objective = Objective(REFERENCE_POSITION)
    print("Building...")
    sim = build_all(config, objective,
                    reference, data, npOde, npSde, sess)
    
    print("Simulation...")
    sim.simulate()
    
    
    print("Saving Data json format")
    data.add_to_list_from_sim(sim)
    data.save_data_json()

    #data.set_file_name(FILE_NAME)
    #print("Saving Data csv format")
    #data.save_data_csv()
 