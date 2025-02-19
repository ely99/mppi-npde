from abc import ABC, abstractmethod
import numpy as np
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import mujoco.viewer
import time
import logging
import traceback
import math
from sbmpc.model import BaseModel, Model, ModelMjx
import sbmpc.settings as settings
from sbmpc.solvers import BaseObjective, SamplingBasedMPC
from typing import Callable, Tuple, Optional, Dict
import json, os, csv #for saving the data



class Dataset:
    def __init__(self):
        self.mj_model = None
        self.mj_data = None
        self.data_list = []  # Lista vuota
        self.data_state = [] 
        self.data_input = []
        self.data_reference = []

        # Variabili della classe
        self.Ts = None
        self.num_parallel_computations = None
        self.time = None
        self.time_horizon = None
        self.file_name = None
        self.directory = None

    def get_data_list(self):
        return self.data_list # observation, 3D ee position

    
    def add_to_list_from_sim(self, sim):
        self.set_state_traj(sim.state_traj)
        self.data_state = self.data_state
        self.data_input = sim.input_traj
        self.data_state = self.data_state.tolist()
        self.data_input = self.data_input.tolist()

    def set_Ts(self, Ts):
        self.Ts = Ts

    def set_num_parallel_computationsm(self, num_parallel_computations):
        self.num_parallel_computations = num_parallel_computations

    def set_time(self, time):
        self.time = time

    def set_time_horizon(self, time_horizon):
        self.time_horizon = time_horizon

    def set_from_config(self, config: settings.Config):
        self.Ts = config.MPC.dt
        self.num_parallel_computations = config.MPC.num_parallel_computations
        self.time = config.sim_iterations*self.Ts
        self.time_horizon = config.MPC.horizon
    
    def set_file_name(self, file_name, directory):
        self.file_name = file_name
        self.directory = directory
    
    def set_mj_model(self, mj_model):
        self.mj_model = mj_model
        
    def set_mj_data(self, mj_data):
        self.mj_data = mj_data
    
    def set_reference(self, reference):
        reference = reference.at[0].set(reference[0] - 1)
        reference = reference.at[3].set(reference[3] + 1)
        self.data_reference = reference.tolist()

    def set_state_traj(self,  state_traj):
        self.data_state = state_traj
        self.data_state[:,0] -=  1
        self.data_state[:,3] +=  1

    def set_input_traj(self, input_traj):
        self.data_input = input_traj

    def save_data_json(self):
        os.makedirs(self.directory, exist_ok=True)
        file_name = os.path.join(self.directory, self.file_name + '.json')

        if None in (self.Ts, self.num_parallel_computations, self.time, self.time_horizon):
            raise ValueError("Set all the variable (Ts, num_sim, time, time_horizon) before using save_data()")
        if any(len(x) == 0 if isinstance(x, list) else x.size == 0 for x in 
            [self.data_state, self.data_input, self.data_reference]):
            raise ValueError("Set all the lists (state, input, reference) before using save_data()")

        new_data = {
            "Ts": self.Ts,
            "num_num_parallel_computations": self.num_parallel_computations,
            "time": self.time,
            "time_horizon": self.time_horizon,
            #"Q": self.Q,
            "reference": self.data_reference,
            "data_list": self.data_list,
            "state_traj": self.data_state,
            "input_traj": self.data_input

        }
        if not os.path.exists(file_name):
            with open(file_name, "w") as file:
                json.dump([new_data], file, indent=4)
        else:      
            with open(file_name, "r+") as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    data = []

                data.append(new_data)

                file.seek(0)
                json.dump(data, file, indent=4)
                file.truncate()

    def save_data_csv(self):
        os.makedirs(self.directory, exist_ok=True)
        file_name = os.path.join(self.directory, self.file_name + '.csv')
        combined_data = np.hstack((self.data_state[:-1], self.data_input))


        with open(file_name, "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerows(combined_data)

    def create_descriptor(self, id_sim, Q, Qf, R):
        os.makedirs(self.directory, exist_ok=True)
        file_name = os.path.join(self.directory, 'descriptor'+str(id_sim) + '.json')
        Q = Q.tolist()
        Qf = Qf.tolist()
        R = R.tolist()
        if None in (self.Ts, self.num_parallel_computations, self.time, self.time_horizon):
            raise ValueError("Set all the variable (Ts, num_parallel_computations, time, time_horizon) before using save_data()")
        if any(len(x) == 0 if isinstance(x, list) else x.size == 0 for x in 
            [self.data_state, self.data_input, self.data_reference]):
            raise ValueError("Set all the lists (state, input, reference) before using save_data()")
        
        new_data = {
            "Ts": self.Ts,
            "num_num_parallel_computations": self.num_parallel_computations,
            "time": self.time,
            "time_horizon": self.time_horizon,
            "reference": self.data_reference,
            "Q": Q,
            "Qf": Qf, 
            "R": R

        }
        with open(file_name, "w") as file:
            json.dump([new_data], file, indent=4)




class Visualizer(ABC):
    def __init__(self):
        self.paused = False
        
    def toggle_paused(self):
        self.paused != self.paused
    
    def get_paused(self):
        return self.paused

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @abstractmethod
    def set_cam_lookat(self, lookat_point: Tuple[float]) -> None:
        pass

    @abstractmethod
    def set_cam_distance(self, distance: float) -> None:
        pass

    @abstractmethod
    def is_running(self) -> bool:
        pass

    @abstractmethod
    def set_qpos(self, qpos) -> None:
        pass


class MujocoVisualizer(Visualizer):
    def __init__(self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData, step_mujoco: bool = True, show_left_ui: bool = False, show_right_ui: bool = False):
        self.mj_data = mj_data
        self.mj_model = mj_model
        self.step_mujoco = step_mujoco
        self.viewer = mujoco.viewer.launch_passive(mj_model,
                                                   mj_data,
                                                   show_left_ui=show_left_ui,
                                                   show_right_ui=show_right_ui,
                                                   key_callback=self.key_callback)

    def key_callback(self, keycode):
        if chr(keycode) == ' ':
            self.toggle_paused()

    def close(self):
        if self.viewer.is_running():
            self.viewer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.viewer.__exit__(exc_type, exc_val, exc_tb)

    def set_cam_lookat(self, lookat_point: Tuple) -> None:
        expected_lookat_size = 3
        actual_size = len(lookat_point)
        if actual_size != expected_lookat_size:
            raise ValueError(f"Invalid look at point. Size should be {expected_lookat_size}, {actual_size} given.")
        self.viewer.cam.lookat = lookat_point

    def set_cam_distance(self, distance: float) -> None:
        self.viewer.cam.distance = distance

    def is_running(self) -> bool:
        return self.viewer.is_running()

    def set_qpos(self, qpos) -> None:
        if self.step_mujoco:
            self.mj_data.qpos = qpos
            mujoco.mj_fwdPosition(self.mj_model, self.mj_data)
        self.viewer.sync()

    def get_body_position(self, body_name):
        end_effector_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        ee_pos = self.mj_data.xpos[end_effector_id]
        return ee_pos.tolist()


def construct_mj_visualizer_from_model(model: BaseModel, scene_path: str):
    mj_model, mj_data = (None, None)
    step_mujoco = True
    if isinstance(model, ModelMjx):
        mj_model = model.mj_model
        mj_data = model.mj_data
    else:
        new_system = ModelMjx(scene_path)
        mj_model = new_system.mj_model
        mj_data = new_system.mj_data
    visualizer = MujocoVisualizer(mj_model, mj_data, step_mujoco=step_mujoco)
    return visualizer

class Simulator(ABC):
    def __init__(self, initial_state, model: BaseModel, controller, data: Dataset, num_iter=1000, visualizer: Optional[Visualizer] = None):
        self.iter = 0
        self.current_state = initial_state
        self.model = model
        self.controller = controller
        self.num_iter = num_iter

        if isinstance(initial_state, (np.ndarray, jnp.ndarray)):
            self.current_state_vec = lambda: self.current_state
        elif isinstance(initial_state, mjx.Data):
            self.current_state_vec = lambda: np.concatenate(
                [self.current_state.qpos, self.current_state.qvel])
        else:
            raise ValueError("""
                        Invalid initial state.
                        """)

        self.state_traj = np.zeros(
            (self.num_iter + 1, self.current_state_vec().size))

        self.state_traj[0, :] = self.current_state_vec()
        self.input_traj = np.zeros((self.num_iter, model.nu))

        self.visualizer = visualizer
        self.data = data # initializa the data class. Look how they managed visualizer since data is a class like him
        self.data.set_mj_model(model.mj_model)
        self.data.set_mj_data(model.mj_data)
        self.paused = False

    @abstractmethod
    def update(self):
        pass

    def simulate(self):
        if self.visualizer is not None:
            try:
                # viewer.set_cam_distance(1.5)
                self.visualizer.set_cam_lookat((0, 0, 0.6))
                list = []
                while self.visualizer.is_running() and self.iter < self.num_iter:
                    if not self.paused:
                        step_start = time.time()
                        self.step()

                        #self.data.add_to_list(self.visualizer.get_body_position("end_effector")) # for the human_arm_model
                        self.visualizer.set_qpos(self.current_state_vec()[
                            :self.model.get_nq()])
                        
                        time_until_next_step = self.controller.dt - \
                            (time.time() - step_start)
                        if time_until_next_step > 0:
                            time.sleep(time_until_next_step)
                
            except Exception as err:
                tb_str = traceback.format_exc()
                logging.error("caught exception below, closing visualizer")
                logging.error(tb_str)
                self.visualizer.close()
                raise
            self.visualizer.close()
        else:
            while self.iter < self.num_iter:
                self.step()

    def step(self):
        self.update()
        self.iter += 1

ROBOT_SCENE_PATH_KEY = "robot_scene_path"

class Simulation(Simulator):
    #data need to be passed as argument
    def __init__(self, initial_state, model, controller, const_reference: jnp.array, data: Dataset, num_iterations: int, visualize: bool = True, visualize_params: Optional[Dict] = None):
        self.const_reference = const_reference
        visualizer = None
        if visualize:
            scene_path = visualize_params.get(ROBOT_SCENE_PATH_KEY, None)
            if visualize_params is None or scene_path is None:
                raise ValueError("if visualizing need to input scene path for mjx")
            visualizer = construct_mj_visualizer_from_model(model, scene_path)

        # data need to be passed as argument
        super().__init__(initial_state, model, controller, data, num_iterations, visualizer)

    def update(self):
        # Compute the optimal input sequence
        time_start = time.time_ns()
        input_sequence = self.controller.command(self.current_state_vec(), self.const_reference, num_steps=1).block_until_ready()
        print("computation time: {:.3f} [ms]".format(1e-6 * (time.time_ns() - time_start)))
        ctrl = input_sequence[0, :].block_until_ready() # take the first input of the sequence

        self.input_traj[self.iter, :] = ctrl

        # Simulate the dynamics
        self.current_state = self.model.integrate(self.current_state, ctrl, self.controller.dt)
        self.state_traj[self.iter + 1, :] = self.current_state_vec()


def build_custom_model(custom_dynamics_fn: Callable, nq: int, nv: int, nu: int, input_min: jnp.array, input_max: jnp.array,
                        q_init: jnp.array, integrator_type: str ="si_euler") -> Tuple[BaseModel, jnp.array, jnp.array]:
    system = Model(custom_dynamics_fn, nq=nq, nv=nv, nu=nu, input_bounds=[input_min, input_max], integrator_type=integrator_type)
    x_init = jnp.concatenate([q_init, jnp.zeros(system.nv, dtype=jnp.float32)], axis=0)
    state_init = x_init
    return system, x_init, state_init


def build_mjx_model(scene_path: str, kinematic: bool = False) -> Tuple[BaseModel, jnp.array, jnp.array]:
    system = ModelMjx(scene_path, kinematic=kinematic)
    q_init = system.data.qpos
    x_init = jnp.concatenate([q_init, jnp.zeros(system.nv, dtype=jnp.float32)], axis=0)
    state_init = system.data
    return system, x_init, state_init

def build_model_from_config(model_type: settings.DynamicsModel, config: settings.Config, custom_dynamics_fn: Optional[Callable] = None):
    if model_type == settings.DynamicsModel.CUSTOM:
        if custom_dynamics_fn is None:
            raise ValueError("for classic dynamics model, a custom dynamics function must be passed. See examples.")
        nq = config.robot.nq
        nv = config.robot.nv
        nu = config.robot.nu
        input_min = config.robot.input_min
        input_max = config.robot.input_max
        q_init = config.robot.q_init
        integrator_type = config.general.integrator_type
        return build_custom_model(custom_dynamics_fn, nq, nv, nu, input_min, input_max, q_init, integrator_type)
    elif model_type == settings.DynamicsModel.MJX:
        return build_mjx_model(config.robot.robot_scene_path, config.robot.mjx_kinematic)
    else:
        raise NotImplementedError

def build_model_and_solver(config: settings.Config, objective: BaseObjective, custom_dynamics_fn: Optional[Callable] = None):
    if config.solver_type != settings.Solver.MPPI:
        raise NotImplementedError
    solver_dynamics_model_setting = config.solver_dynamics
    system, solver_x_init, sim_state_init = build_model_from_config(solver_dynamics_model_setting, config, custom_dynamics_fn)
    solver = SamplingBasedMPC(system, objective, config)
    return system, solver

def build_all(config: settings.Config, objective: BaseObjective, 
              reference: jnp.array, data: Dataset, npOde, npSde,  sess,
              custom_dynamics_fn: Optional[Callable] = None):
    system, x_init, state_init = (None, None, None)
    solver_dynamics_model_setting = config.solver_dynamics
    sim_dynamics_model_setting = config.sim_dynamics

    solver_dynamics_model, sim_dynamics_model = (None, None)
    solver_x_init, sim_state_init = (None, None)
    if solver_dynamics_model_setting == sim_dynamics_model_setting:
        system, solver_x_init, sim_state_init = build_model_from_config(solver_dynamics_model_setting, config, custom_dynamics_fn)
        solver_dynamics_model = system
        sim_dynamics_model = system
    else:
        system, solver_x_init, _ = build_model_from_config(solver_dynamics_model_setting, config, custom_dynamics_fn)
        solver_dynamics_model = system
        sim_dynamics_model, _, sim_state_init = build_model_from_config(sim_dynamics_model_setting, config, custom_dynamics_fn)

    if config.solver_type != settings.Solver.MPPI:
        raise NotImplementedError

    solver = SamplingBasedMPC(solver_dynamics_model, objective, config, npOde, npSde, sess)
    
    # dummy for jitting
    input_sequence = solver.command(solver_x_init, reference, False).block_until_ready()
    visualize = config.general.visualize
    visualizer_params = {ROBOT_SCENE_PATH_KEY: config.robot.robot_scene_path}

    # Setup and run the simulation
    num_iterations = config.sim_iterations

    sim = Simulation(sim_state_init, sim_dynamics_model, solver, reference, data, num_iterations, visualize, visualizer_params)
    return sim

def Generator_feasible_desired_state(half_len, x_com, y_com, theta) ->jnp.array:
    theta = np.deg2rad(theta)
    delta_x = half_len * math.cos(theta)
    delta_y = half_len*math.sin(theta)
    x_r = x_com + delta_x
    y_r = y_com + delta_y
    x_l = x_com - delta_x
    y_l = y_com - delta_y
    return jnp.array([x_l + half_len, y_l, 0., x_r - half_len, y_r, 0., 0., 0., 0., 0., 0., 0.], dtype=jnp.float32)