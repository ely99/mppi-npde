# this file is useful for plotting the mujoco environment
import mujoco
import mujoco.viewer
import os, sys

# Path to the XML model file
script_dir = (os.path.join(os.path.dirname(__file__), '..'))
xml_path = os.path.join(script_dir, "H-B-D/models/huma-bar-drone.xml") 
#xml_path = os.path.join(script_dir, "H-B-D/models/test1.xml") 

# Load the MuJoCo model
model = mujoco.MjModel.from_xml_path(xml_path)

# Create a simulation data object
data = mujoco.MjData(model)


# Launch the viewer and visualize the model with joint control enabled
with mujoco.viewer.launch(model, data) as viewer:
    while viewer.is_running():
        # Advance simulation one step
        mujoco.mj_step(model, data)



        # Allow interaction via viewer
        viewer.sync()
