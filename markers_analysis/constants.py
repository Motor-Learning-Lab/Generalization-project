# This file is for the constants used all over the program.
# Rather than making these parts modular, I've found that it's clearer and simpler to make
# constants of the experiment global and universally accessible.

import os

import numpy as np

################
# File locations
################

data_basepath = os.sep.join(["..", "data"])
table_data_basename = "white_ball_hit "
markers_data_basename = "white_ball_hit "


def table_data_path_fn(basepath, date, id):
    return os.sep.join([basepath, f"subject_{id}", date, "table"])


def markers_data_path_fn(basepath, date, id):
    return os.sep.join([basepath, f"subject_{id}", date, "markers"])


results_basepath = os.sep.join(["..", "results"])


################################
# Label data and table geometry
################################

likelihood_threshold = 0.9

# These are made up right now (or best guesses).
# They are largely supplanted by the calibration using the table dots in DeepLabCut
x_pixels_per_cm = 3.5  # This is a wild guesss. We need real numbers here
y_pixels_per_cm = 2.5
# This may change for a better camera
frames_per_sec = 30.0  # 30 frames per second

# ### Constants for determining initial position
moving_ball_speed_threshold = 10  # cm /s for ball not moving
correct_position_threshold = 2  # cm for ball being where it is supposed to be

# ### Sizes of table and balls
ball_radius = 4.5
# ### Locations of the dots
table_length = 162.3
table_width = 74.5
dot_setback = 5.3

# ### Location of the dots on the table, used for calibration

dots_cm = dict()

mid_length = table_length / 2
mid_width = table_width / 2

dots_cm["Dot_Dn_L"] = np.array([mid_length - 61.5, -5.2])
dots_cm["Dot_Dn_ML"] = np.array([mid_length - 20.5, -5.3])
dots_cm["Dot_Dn_MR"] = np.array([mid_length + 20.5, -5.5])
dots_cm["Dot_Dn_R"] = np.array([mid_length + 61.8, -5.2])
dots_cm["Dot_Up_L"] = np.array([mid_length - 61.5, table_width + 4.6])
dots_cm["Dot_Up_ML"] = np.array([mid_length - 20.5, table_width + 5.2])
dots_cm["Dot_Up_MR"] = np.array([mid_length + 20.5, table_width + 5.3])
dots_cm["Dot_Up_R"] = np.array([mid_length + 61.5, table_width + 5.4])
dots_cm["Dot_Lt_D"] = np.array([-5.2, mid_width - 20.5])
# dots_cm["Dot_Lt_M"] we don't include this because it is generally hidden by the player
dots_cm["Dot_Lt_U"] = np.array([-5.4, mid_width + 20.8])
dots_cm["Dot_Rt_D"] = np.array([table_length + 5.5, mid_width - 20.5])
dots_cm["Dot_Rt_M"] = np.array([table_length + 5.5, mid_width])
dots_cm["Dot_Rt_U"] = np.array([table_length + 5.3, mid_width + 20.5])


# ### Window for predicting the ball will hit the edge
table_hit_window = 0.1

# ### Order of the points in the table. Important for drawing and checking
# ### whether the ball hit the end.
table_order = [
    "Tbl_Dn_L",
    "Tbl_Dn_R",
    "Tbl_Rt_D",
    "Tbl_Rt_U",
    "Tbl_Up_R",
    "Tbl_Up_L",
    "Tbl_Lt_U",
    "Tbl_Lt_D",
]




####################################
# Plotting defaults
####################################

marker_freq_lowpass = 15



####################################
# Plotting defaults
####################################

color_mapping = {
    "cue_ball": "white",
    "object_ball": "purple",
    "target_ball": "red",
    "cue": "yellow",
    "table": "darkcyan",
    "mat": "skyblue",
    "dots": "brown",
}


####################################
# Trial definition constants
####################################

# Start of shot
start_positions_time = (
    0.5  # s How long the ball need to be in the start position to start a trial
)
window_of_stillness = 0.2  # s How long before the end of the start conditions to cut for a little stillness at the beginning of the trial

minimum_trial_gap = 0.1  # s At least 100 ms between trials
still_ball_time = 0.1  # s At least 100 ms for a ball to be considered stopped
moving_ball_time = 0.1  # s At least 100 ms for a ball to be considered moving
table_hit_window = 10  # cm Window in which ball is about to hit the edge that is considered "close enough"
nan_run_time = 0.05  # How long a run of nans has to be to end a trial


####################################
# Constants for cutting marker data
####################################

cut_times = [-1.0, 1.0]


####################################
# List of marker names
####################################

# Body
body_markers = [
    "R_GR",  # Right hip (Greater trocanter, I thnk)
    "L_GR",  # Left hip
    "L5/S1",  # Lower back (Fith lumbar / first sacral spine)
    "R_SAE",  # Right shoulder
    "L_SAE",  # Left shoulder
    "R_ELB",  # Right elbow (average of lateral and medial)
    "R_ELB_M",  # Right elbow medial
    "R_ELB_L",  # Right elbow lateral
    "L_ELB",  # Left elbow (average of lateral and medial)
    "L_ELB_M",  # Left elbow medial
    "L_ELB_L",  # Left elbow lateal
    "R_W",  # Right wrist (average of lateral and medial)
    "R_W_L",  # Right wrist lateral
    "R_W_M",  # Right wrist medial
    "L_W",  # Left wrist (average of lateral and medial)
    "L_W_M",  # Left wrist medial
    "L_W_L",  # Left wrist lateral
    "L_H_L",  # Left hand lateral (pnky knuckle)
    "L_H_M",  # Left hand medial (pointe knuckle)
    "R_H_L",  # Right hand lateral
    "R_H_M",  # Right hand medial
]

# Cue
cue_markers = [
    "FQ",  # Front of cue
    "BQ",  # Back of cue
]
