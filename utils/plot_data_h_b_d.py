"""This function import data.json file and plot all the data. the file MUST be generated from the 
human arm simulation otherwhise it is useles.
PLEASE SELECT THE CORRECT INDEX OF THE SIMULATION YOU WANT TO PLOT, """

import json
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

SIMULATION_ID = 6
file_name = os.path.join('SIMULATION_DATA/', 'some_sim'+'.json') 
#file_name = os.path.join('H-B-D/Generate_Dataset_HBD/DATASET/SIM0', 'data1'+'.json')

try:
    with open(file_name, "r") as file:
        data = json.load(file)
                
except FileNotFoundError:
    print(f"Error: File {file_name} not found.")
    sys.exit(1)
except json.JSONDecodeError:
    print("Error: No valid Json data")
    sys.exit(1)

if SIMULATION_ID < 0 :
    print(f"Error: SIMULATION_ID {SIMULATION_ID} is out of range.")
    sys.exit(1)

simulation_data = data[SIMULATION_ID] # the index defines the number of the simulation, look inside the file how many simulation are stored
flag_3d = False

Ts = simulation_data.get("Ts")  # dt
num_sim = simulation_data.get("num_sim")  # num_parallel_computations
time = simulation_data.get("time")  # duration of the simulation
time_horizon = simulation_data.get("time_horizon")

reference = np.array(simulation_data.get("reference"))
# move from joint space to world space
#reference[0] -= 1
#reference[3] += 1


data_list = np.array(simulation_data.get("data_list"))

state_traj = np.array(simulation_data.get("state_traj"))
input_traj = np.array(simulation_data.get("input_traj"))
weight = np.array([[1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0]])


# PLOT THE DATA
common_length = min(state_traj.shape[0], input_traj.shape[0])

time_vect = Ts * jnp.arange(common_length)
rep_ref = np.tile(reference, (common_length,1))
state_traj_truncated = state_traj[:common_length] 
input_traj_truncated = input_traj[:common_length]
fig, axes = plt.subplots(7, 2, figsize=(15, 18), sharex=True)
fig.suptitle(f"Simulation {SIMULATION_ID} - SIMPLE ARM. Body frame !!!! ")
offset = np.ones(common_length) 

# NOW FROM JOINT SPACE TO WORLD SPACE

#state_traj_truncated[:, 0] -= offset
#state_traj_truncated[:, 3] += offset


err = np.abs(state_traj_truncated - rep_ref) 
err = err@weight.T
# Row 1: x position (indices 0 and 3 of state_traj_truncated)
axes[0, 0].plot(time_vect, state_traj_truncated[:, 0], label="X Left [m]", color="green")
axes[0, 0].plot(time_vect, jnp.ones_like(time_vect) * reference[0], label="Desired", color="black", linestyle="--", linewidth=2)
axes[0, 0].set_ylabel("Pos X Left [m]")
axes[0, 0].grid()

axes[0, 1].plot(time_vect, state_traj_truncated[:, 3], label="X Right [m]", color="red")
axes[0, 1].plot(time_vect, jnp.ones_like(time_vect) * reference[3], label="Desired", color="black", linestyle="--", linewidth=2)
axes[0, 1].set_ylabel("Pos X Right [m]")
axes[0, 1].grid()

# Row 2: x velocity (indices 6 and 9 of state_traj_truncated)
axes[1, 0].plot(time_vect, state_traj_truncated[:, 6], label="X Left Dot [m/s]", color="blue")
axes[1, 0].plot(time_vect, jnp.ones_like(time_vect) * reference[6], label="Desired", color="black", linestyle="--", linewidth=2)
axes[1, 0].set_ylabel("Vel X Left [m/s]")
axes[1, 0].grid()

axes[1, 1].plot(time_vect, state_traj_truncated[:, 9], label="X Right Dot [m/s]", color="purple")
axes[1, 1].plot(time_vect, jnp.ones_like(time_vect) * reference[9], label="Desired", color="black", linestyle="--", linewidth=2)
axes[1, 1].set_ylabel("Vel X Right [m/s]")
axes[1, 1].grid()

# Row 3: x forces (indices 0 and 2 of input_traj_truncated)
axes[2, 0].plot(time_vect, input_traj_truncated[:, 0], label="Input [Nm]", color="cyan")
axes[2, 0].set_ylabel("Force X Left [Nm]")
axes[2, 0].grid()

axes[2, 1].plot(time_vect, input_traj_truncated[:, 2], label="Input [Nm]", color="magenta")
axes[2, 1].set_ylabel("Force X Right [Nm]")
axes[2, 1].grid()

# Row 4: y position (indices 1 and 4 of state_traj_truncated)
axes[3, 0].plot(time_vect, state_traj_truncated[:, 1], label="Y Left [m]", color="green")
axes[3, 0].plot(time_vect, jnp.ones_like(time_vect) * reference[1], label="Desired", color="black", linestyle="--", linewidth=2)
axes[3, 0].set_ylabel("Pos Y Left [m]")
axes[3, 0].grid()

axes[3, 1].plot(time_vect, state_traj_truncated[:, 4], label="Y Right [m]", color="red")
axes[3, 1].plot(time_vect, jnp.ones_like(time_vect) * reference[4], label="Desired", color="black", linestyle="--", linewidth=2)
axes[3, 1].set_ylabel("Pos Y Right [m]")
axes[3, 1].grid()

# Row 5: y velocity (indices 7 and 10 of state_traj_truncated)
axes[4, 0].plot(time_vect, state_traj_truncated[:, 7], label="Y Left Dot [m/s]", color="blue")
axes[4, 0].plot(time_vect, jnp.ones_like(time_vect) * reference[7], label="Desired", color="black", linestyle="--", linewidth=2)
axes[4, 0].set_ylabel("Vel Y Left [m/s]")
axes[4, 0].grid()

axes[4, 1].plot(time_vect, state_traj_truncated[:, 10], label="Y Right Dot [m/s]", color="purple")
axes[4, 1].plot(time_vect, jnp.ones_like(time_vect) * reference[10], label="Desired", color="black", linestyle="--", linewidth=2)
axes[4, 1].set_ylabel("Vel Y Right [m/s]")
axes[4, 1].grid()

# Row 6: y forces (indices 1 and 3 of input_traj_truncated)
axes[5, 0].plot(time_vect, input_traj_truncated[:, 1], label="Input [Nm]", color="cyan")
axes[5, 0].set_ylabel("Force Y Left [Nm]")
axes[5, 0].grid()

axes[5, 1].plot(time_vect, input_traj_truncated[:, 3], label="Input [Nm]", color="magenta")
axes[5, 1].set_ylabel("Force Y Right [Nm]")
axes[5, 1].grid()

# Row 7: y forces (indices 1 and 3 of input_traj_truncated)
axes[6, 0].plot(time_vect, err[:, 0], label="err [m]", color="orange")
axes[6, 0].plot(time_vect, jnp.ones_like(time_vect) * 0, label="Desired", color="black", linestyle="--", linewidth=2)
axes[6, 0].set_ylabel("Position error [m]")
axes[6, 0].grid()

axes[6, 1].grid()
axes[6, 1].plot(time_vect, err[:, 1], label="err [m/s]", color="blue")
axes[6, 1].plot(time_vect, jnp.ones_like(time_vect) * 0, label="Desired", color="black", linestyle="--", linewidth=2)
axes[6, 1].set_ylabel("Velocity error [m/s]")

# Plot 3D trajectories for left and right components in the same graph


# Creazione della figura e asse 3D
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlim(-3., 3.)
ax.set_ylim(-3., 3.)
ax.view_init(elev=90, azim=0)
# Etichette e titolo
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.set_title("3D Trajectories")

# Inizializzazione delle linee (vuote)
left_traj, = ax.plot([], [], [], color="green", label="Trajectory Left", alpha=0.7)
right_traj, = ax.plot([], [], [], color="red", label="Trajectory Right", alpha=0.7)
current_line = None  # Per la linea viola

# Aggiunta dei punti di inizio
ax.scatter(state_traj_truncated[0, 0], state_traj_truncated[0, 1], state_traj_truncated[0, 2], 
           color="green", s=5, label="Start Left")
ax.scatter(state_traj_truncated[0, 3], state_traj_truncated[0, 4], state_traj_truncated[0, 5], 
           color="red", s=5, label="Start Right")
ax.scatter(reference[0], reference[1], 0, color="yellow", s=20, label="End Left")
ax.scatter(reference[3], reference[4], 0, color="orange", s=20, label="End Right")
ax.legend()

if(flag_3d):
    # Ciclo per mostrare l'evoluzione temporale
    for i in range(len(state_traj_truncated)):
        # Aggiornamento delle traiettorie verdi e rosse (storico)
        left_traj.set_data(state_traj_truncated[:i+1, 0], state_traj_truncated[:i+1, 1])
        left_traj.set_3d_properties(jnp.zeros(i+1))  # Z fisso a zero
        
        right_traj.set_data(state_traj_truncated[:i+1, 3], state_traj_truncated[:i+1, 4])
        right_traj.set_3d_properties(jnp.zeros(i+1))  # Z fisso a zero
        
        # Rimuovi la linea viola precedente, se esiste
        if current_line is not None:
            current_line.pop(0).remove()

        # Linea viola corrente
        current_line = ax.plot(
            [state_traj_truncated[i, 0], state_traj_truncated[i, 3]],  # X
            [state_traj_truncated[i, 1], state_traj_truncated[i, 4]],  # Y
            [0, 0],  # Z fisso a zero
            color="purple", linestyle="-", linewidth=2
        )
        plt.pause(0.01)


else:
    ax.plot(state_traj_truncated[:, 0], state_traj_truncated[:, 1], 0, 
        label="Trajectory Left", color="green")
    ax.scatter(state_traj_truncated[0, 0], state_traj_truncated[0, 1], 0, 
            color="green", s=10, label="Start Left")
    ax.plot(state_traj_truncated[:, 3], state_traj_truncated[:, 4], 0, 
            label="Trajectory Right", color="red")
    ax.scatter(state_traj_truncated[0, 3], state_traj_truncated[0, 4], 0, 
            color="red", s=10, label="Start Right")
    """
    for i in range(len(state_traj_truncated)):
        ax.plot([state_traj_truncated[i, 0], state_traj_truncated[i, 3]],  # X
                [state_traj_truncated[i, 1], state_traj_truncated[i, 4]],  # Y
                [0, 0],  # Z
                color="purple", linestyle="--", linewidth=1) """

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("3D Trajectories")



plt.show()
