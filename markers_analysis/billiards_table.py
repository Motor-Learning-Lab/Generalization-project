# # Basic processing of ball videos
# The videos include 3 balls in different colors
# (the white ball is the cue ball, the purple is the object ball,
# and the red is the target ball).
# The videos also have the tip of the cue and a point on the back of the cue identified

import numpy as np
import pandas as pd
import cv2 as cv
import copy as copy

# from numba import jit


from matplotlib import pyplot as plt
from matplotlib import animation as animation

import kineticstoolkit as ktk

from . import constants as consts
from . import math as my_math


def load_dataframe(data_file):
    """
    Read a DataFrame from either CSV or HDF5 file.

    Parameters:
    - file_path (str): The path to the file.

    Returns:
    - pd.DataFrame: The loaded DataFrame.
    """
    if data_file.endswith(".csv"):
        # Read from CSV
        df = pd.read_csv(data_file, index_col=0, header=list(range(3)))
    elif data_file.endswith(".h5") or data_file.endswith(".hdf5"):
        # Read from HDF5
        df = pd.read_hdf(data_file)
    else:
        # Unsupported file format
        raise ValueError(f"Unsupported file format: {data_file}")

    return df


def load_data(
    data_file,
    likelihood_threshold=consts.likelihood_threshold,
    column_mapping={
        "white": "cue_ball",
        "purple": "object_ball",
        "red": "target_ball",
        "start_tip": "start_cue",
        "end_tip": "end_cue",
    },
    x_pixels_per_cm=consts.x_pixels_per_cm,
    y_pixels_per_cm=consts.y_pixels_per_cm,
    frames_per_sec=consts.frames_per_sec,
):
    data = load_dataframe(data_file)
    data = data.droplevel(0, axis=1)

    data.columns.names = ["objects", "coords"]
    if column_mapping:
        data = data.rename(column_mapping, axis=1, level=0)

    cols = data.columns.get_level_values(0).unique()
    for c in cols:
        # Create a mask for rows where 'p' for the current color
        # is less than the threshold
        mask = data[(c, "likelihood")] < likelihood_threshold
        # Set 'x' and 'y' to NaN for the rows where the mask is True
        data.loc[mask, (c, ["x", "y"])] = np.nan

    data = data.drop(labels="likelihood", axis=1, level="coords")

    idx = pd.IndexSlice
    data.loc[:, idx[:, "x"]] /= x_pixels_per_cm
    data.loc[:, idx[:, "y"]] /= y_pixels_per_cm

    data["t"] = data.index / frames_per_sec
    return data


def _update_lims(xlims, ylims, data):
    for s_name, s in data.items():
        s = np.atleast_2d(s)

        x_min = np.nanmin(s[:, 0]) if np.any(np.isfinite(s[:, 0])) else xlims[0]
        x_max = np.nanmax(s[:, 0]) if np.any(np.isfinite(s[:, 0])) else xlims[1]

        y_min = np.nanmin(s[:, 1]) if np.any(np.isfinite(s[:, 1])) else ylims[0]
        y_max = np.nanmax(s[:, 1]) if np.any(np.isfinite(s[:, 1])) else ylims[1]

        xlims = [np.min([xlims[0], x_min]), np.max([xlims[1], x_max])]
        ylims = [np.min([ylims[0], y_min]), np.max([ylims[1], y_max])]

    return xlims, ylims


# This creates a video animation of the identified markers.
#
# - The balls are given a 'trail' of 1.5 seconds so that hey can be more easily seen.
# - The table has a blue background so we can use a white ball for the cue.
def create_video(
    data_dict,
    file_name,
    keep_time=None,
    do_balls=True,
    do_cue=True,
    do_table=True,
    do_dots=False,
    color_mapping=consts.color_mapping,
    table_order=consts.table_order,
    trail_length_sec=1.5,
    xlims=None,
    ylims=None,
    frame_titles=None,
    frame_label=None,
    frame_titles_format=None,
):
    print(file_name)

    table = data_dict["table"]
    ts = data_dict["ts"]
    if keep_time:
        ts = copy.deepcopy(ts)
        keep_time[1] = np.min([keep_time[1], ts.time[-1]])
        keep_time[0] = np.max([ts.time[0], keep_time[0]])
        ts = ts.get_ts_between_times(keep_time[0], keep_time[1])

    balls = ts.get_subset([k for k, v in ts.data_info.items() if v["Type"] == "Ball"])
    cue = ts.get_subset([k for k, v in ts.data_info.items() if v["Type"] == "Cue"])

    if frame_titles is None:
        frame_titles = ts.time
    if frame_label is None:
        frame_label = "Time"
    if frame_titles_format is None:
        frame_titles_format = ".2f"

    figure, ax = plt.subplots()

    lines = dict()

    default_xlims = [np.inf, -np.inf]
    default_ylims = [np.inf, -np.inf]

    if do_balls and balls:
        not_in_color_mapping = [b for b in balls.data.keys() if b not in color_mapping]
        if not_in_color_mapping:
            s = "balls_list has balls that are not in the color mapping: "
            s = s + f"{not_in_color_mapping}"
            raise ValueError(s)

        trail_frames = int(np.ceil(trail_length_sec * consts.frames_per_sec))

        for b_name, b in balls.data.items():
            x0 = b[0, 0]
            y0 = b[0, 1]
            c = color_mapping[b_name]
            (lines[b_name],) = ax.plot(x0, y0, color=c)

        default_xlims, default_ylims = _update_lims(
            default_xlims, default_ylims, balls.data
        )
    else:
        do_balls = False

    print(f"balls: {default_xlims} {default_ylims}")

    if do_cue and cue:
        if "cue" not in color_mapping:
            raise ValueError("Cue must be assigned a color in color mapping")

        cuex = np.array([v[0, 0] for v in cue.data.values()])
        cuey = np.array([v[0, 1] for v in cue.data.values()])
        (lines["cue"],) = ax.plot(
            cuex,
            cuey,
            color=color_mapping["cue"],
        )

        default_xlims, default_ylims = _update_lims(
            default_xlims, default_ylims, cue.data
        )
    else:
        do_cue = False

    print(f"cue: {default_xlims} {default_ylims}")

    if do_table and table:
        if "table" not in color_mapping:
            raise ValueError("Table must be assigned a color in color mapping")

        tablex = np.array([table[v][0] for v in table_order])
        tabley = np.array([table[v][1] for v in table_order])
        (lines["table"],) = ax.plot(
            np.append(tablex, tablex[0]),
            np.append(tabley, tabley[0]),
            color=color_mapping["table"],
        )

        default_xlims, default_ylims = _update_lims(default_xlims, default_ylims, table)
    else:
        do_table = False

    print(f"table: {default_xlims} {default_ylims}")

    if do_dots and "dots" in data_dict.keys():
        dots = data_dict["dots"]

        if "dots" not in color_mapping:
            raise ValueError("Dots must be assigned a color in color mapping")

        dotsx = np.array([dots[v][0] for v in dots.keys()])
        dotsy = np.array([dots[v][1] for v in dots.keys()])
        (lines["dots"],) = ax.plot(
            dotsx,
            dotsy,
            color=color_mapping["dots"],
            markersize=20,
            marker=".",
            linestyle="",
        )

        default_xlims, default_ylims = _update_lims(default_xlims, default_ylims, dots)
    else:
        do_dots = False

    print(f"dots: {default_xlims} {default_ylims}")

    if xlims is None:
        xlims = default_xlims
    if ylims is None:
        ylims = default_ylims

    # Setting limits for x and y axis
    ax.set_xlim(xlims[0], xlims[1])
    ax.set_ylim(ylims[0], ylims[1])
    ax.axis("equal")
    ax.set_facecolor(color_mapping["mat"])

    def animation_function(i):
        if do_balls:
            for b_name, b in balls.data.items():
                indexes = slice(np.max([i - trail_frames, 0]), i)
                lines[b_name].set_xdata([b[indexes, 0]])
                lines[b_name].set_ydata([b[indexes, 1]])

        if do_cue:
            cuex = np.array([v[i, 0] for v in cue.data.values()])
            cuey = np.array([v[i, 1] for v in cue.data.values()])
            lines["cue"].set_xdata(cuex)
            lines["cue"].set_ydata(cuey)

        ax.set_title(f"{frame_label}: {frame_titles[i]:{frame_titles_format}}")

        return (lines,)

    ani = animation.FuncAnimation(
        figure, func=animation_function, frames=range(len(ts.time)), interval=50
    )

    ani.save(filename=file_name, writer="ffmpeg")

    plt.close()


# ##### Calculate speed of ball
#
# 1. Interpolate the missing values to prevent spikes in the derivative
# 2. Fit a spline to both x and y
# 3. Take the derivative of the fit spline
# @jit
def get_velocities(balls):
    vels_1 = ktk.filters.deriv(balls)  # This shortens the vector by 1!!
    vels_1 = ktk.filters.median(vels_1, window_length=5)

    vels_data = {k: np.vstack([v, v[-1, :]]) for k, v in vels_1.data.items()}
    vels_data_info = vels_1.data_info
    for k, v in vels_data_info.items():
        v["Units"] = v["Units"] + "/s"
        v["Type"] = v["Type"] + " Velocity"
        vels_data_info[k] = v

    vels = ktk.TimeSeries(
        time=balls.time,
        time_info=balls.time_info,
        data=vels_data,
        data_info=vels_data_info,
    )
    return vels


def get_speed_from_velocity(ball_vels):
    speed_data = {k: np.linalg.norm(v, axis=1) for k, v in ball_vels.data.items()}

    speed_data_info = ball_vels.data_info
    for k, v in ball_vels.data_info.items():
        v["Type"] = v["Type"].replace("Velocity", "Speed")
        speed_data_info[k] = v

    ball_speeds = ktk.TimeSeries(
        time=ball_vels.time,
        time_info=ball_vels.time_info,
        data=speed_data,
        data_info=speed_data_info,
    )
    return ball_speeds


def get_speeds(balls):
    ball_vels = get_velocities(balls)
    ball_speeds = get_speed_from_velocity(ball_vels)
    return ball_speeds


"""
def get_roll_angle(ball, ball_vel, target):
    vec_to_target = target - ball
    vec_target_magnitudes = np.linalg.norm(vec_to_target, axis=1)
    vec_to_target = vec_to_target / vec_target_magnitudes[:, np.newaxis]
    perpendicular_to_target = np.stack(
        [-vec_to_target[:, 1], vec_to_target[:, 0]], axis=1
    )

    vel_magnitudes = np.linalg.norm(ball_vel, axis=1)
    unit_vels = ball_vel / vel_magnitudes[:, np.newaxis]

    vel_dot_target = np.sum(vec_to_target * unit_vels, axis=1)
    vel_dot_perpendicular = np.sum(perpendicular_to_target * unit_vels, axis=1)
    vel_angle = np.arctan2(vel_dot_perpendicular, vel_dot_target)
    return vel_angle
"""


def get_roll_angle(ball, target):
    x = ball[:, 0]
    y = ball[:, 1]
    A = np.vstack([x, np.ones(len(x))]).T

    nan_rows = np.isnan(A).any(axis=1) | np.isnan(y)
    A = A[~nan_rows]
    y = y[~nan_rows]

    m, c = np.linalg.lstsq(A, y, rcond=1e-5)[0]

    yhat = m * x + c

    target = target.mean(axis=0)

    ab = np.array(target) - np.array([x[0], yhat[0]])
    ac = np.array([x[-1], yhat[-1]]) - np.array([x[0], yhat[0]])

    # Calculate the dot product
    dot_product = np.dot(ab, ac)
    det_product = np.linalg.det([ab, ac])

    angle_rad = np.arctan2(det_product, dot_product)

    return angle_rad


def is_moving(speeds, speed_threshold=consts.moving_ball_speed_threshold):
    is_moving_data = {k: speed_threshold < speed for k, speed in speeds.data.items()}

    is_moving_data_info = speeds.data_info
    for k, v in speeds.data_info.items():
        v["Type"] = v["Type"].replace("Speed", "Moving")
        v["Units"] = "None"
        is_moving_data_info[k] = v

    balls_moving = ktk.TimeSeries(
        time=speeds.time,
        time_info=speeds.time_info,
        data=is_moving_data,
        data_info=is_moving_data_info,
    )
    return balls_moving

    return is_moving


def is_in_position(
    objects,
    correct_object_positions=None,
    positioned_objects_list=None,
    position_threshold=consts.correct_position_threshold,
):
    # If correct balls not specified, use modes
    if correct_object_positions is None:
        # If which balls not specified, checkk for all
        if positioned_objects_list is None:
            correct_object_positions = my_math.find_modes_dict(objects)
        else:
            correct_object_positions = my_math.find_modes_dict(
                objects, label_list=positioned_objects_list
            )

    # If which balls not specified, use the ones for which we have a correct position
    if positioned_objects_list is None:
        positioned_objects_list = correct_object_positions.keys()

    # Each ball we care about must have a correct position
    not_in_list = [
        b for b in correct_object_positions if b not in positioned_objects_list
    ]
    if not_in_list:
        raise ValueError(
            f"The following appear in correct_balls_positions list but not in positioned_balls_list: {not_in_list}"  # noqa: E501
        )  # noqa: E501

    all_ok = None
    for b in positioned_objects_list:
        correct_position = correct_object_positions[b]
        dist2 = my_math.dist2(objects[b], correct_position)
        b_ok = dist2 < position_threshold**2
        if all_ok is None:
            all_ok = b_ok
        else:
            all_ok = np.logical_and(all_ok, b_ok)
    return all_ok, correct_object_positions


def is_in_order(object1, object2, dim=0):
    return object1[:, dim] < object2[:, dim]


def get_table_edges(table, table_order=consts.table_order):
    point_names = [p for p in table_order if p in table.keys()]
    edges = list(
        [
            np.vstack([table[k1], table[k2]])
            for k1, k2 in zip(point_names, point_names[1:] + [point_names[0]])
        ]
    )
    return edges


def hit_table_edge(
    t,
    ball,
    ball_vel,
    table,
    table_hit_window=consts.table_hit_window,
    return_distance=False,
):
    # A list of line segments giving the table edges
    table_edges = get_table_edges(table)

    dt = t[1] - t[0]

    all_seg_dists = []
    all_seg_hits = []
    for edge in table_edges:
        # The distance from the ball to one line segment over time
        seg_dists = my_math.dist_segment(ball, edge[0, :], edge[1, :])
        # d_seg_dists = my_math.d_dist_segment(ball, ball_vel, edge[0,:], edge[1,:])
        all_seg_dists.append(seg_dists)

        hitting = seg_dists < table_hit_window
        # bouncing_off = (d_seg_dists[:-1] < 0) & (d_seg_dists[1:] > 0)
        # bouncing_off = np.concatenate([bouncing_off, [True]])
        seg_hits = hitting
        # seg_hits = hitting & bouncing_off

        # seg_hits = seg_dists < table_hit_dist
        all_seg_hits.append(seg_hits)

    all_seg_dists = np.vstack(all_seg_dists)
    all_seg_hits = np.vstack(all_seg_hits).any(axis=0)
    min_dist = np.min(all_seg_dists, axis=0)
    if return_distance:
        return all_seg_hits, min_dist
    else:
        return all_seg_hits


# #### Find runs of ok position
#
# Chat-GPT helped me.
#
# - Every time it's ok and wasn't in the last frame is the start of a run.
# - Every time it's ok and isn't in the next frame is the end of a run
#
# We keep only runs that are at least 0.5 seconds.
def find_runs(t, condition, run_threshold_seconds):
    # Find the indices where True values start and end
    start_indices = np.where(condition & ~np.roll(condition, 1))[0]
    end_indices = np.where(condition & ~np.roll(condition, -1))[0]

    # Handle the case where the first end comes before the start
    if start_indices.size > 0:
        if end_indices.size > 0:
            if end_indices[0] < start_indices[0]:
                end_indices = end_indices[1:]
            # Handle the case where the run ends at the last element
            if (not end_indices.size > 0) or end_indices[-1] < start_indices[-1]:
                end_indices = np.append(end_indices, len(t) - 1)
        else:
            end_indices = start_indices

        # Calculate the lengths of the runs
        start_times = t[start_indices]
        end_times = t[end_indices]

        run_durations = end_times - start_times
        long_runs = run_durations >= run_threshold_seconds
    else:
        start_indices = np.array([], dtype=int)
        end_indices = np.array([], dtype=int)
        start_times = np.array([], dtype=float)
        end_times = np.array([], dtype=float)
        run_durations = np.array([], dtype=float)
        long_runs = np.array([], dtype=int)

    df_indices = pd.DataFrame(
        {
            "start_indices": start_indices[long_runs],
            "end_endices": end_indices[long_runs],
        }
    )
    df_times = pd.DataFrame(
        {
            "start_times": start_times[long_runs],
            "end_times": end_times[long_runs],
            "run_durations": run_durations[long_runs],
        }
    )

    return pd.concat(
        [df_indices, df_times],
        join="inner",
        axis=1,
    )


# ### Cut the data into shots
#
# A shot goes from a bit (0.2 sec) before the end of everything being in position
# to a bit before you get to the NaNs (0.05 sec).
#
# There are a number of special cases to handle:
#
# - A shot that starts very near the beginning of the file
# (so we rest it to start at the beginning)
# - A shot that starts after the last run of NaNs
# (and so goes to the end of the file)
# - A shot that doesn't really end in a run of NaNs
# (and so it ends before the next shot begins)
#
# All of the shots are collected into a dataframe that we can use for
# indexing when we loop over the data.
def find_epochs(t, epoch_start_times, epoch_end_times, minimum_gap_time=0.0):
    if isinstance(t, pd.Series):
        t = t.values
    if isinstance(epoch_start_times, pd.Series):
        epoch_start_times = epoch_start_times.values
    if isinstance(epoch_end_times, pd.Series):
        epoch_end_times = epoch_end_times.values

    next_start_time = epoch_start_times[1:]
    next_start_time = np.append(next_start_time, np.nan)

    epoch_list = []
    for start_time, next_start in zip(epoch_start_times, next_start_time):
        end_times_after = epoch_end_times[epoch_end_times > start_time]

        if np.any(end_times_after):
            end_time = end_times_after[0]
            if end_time > next_start:
                end_time = next_start - minimum_gap_time
        else:
            end_time = t[-1]

        start_index = np.max(len(t[t <= start_time]) - 1, 0)
        end_index = len(t[t <= end_time])

        epoch_list.append(
            {
                "start_index": start_index,
                "end_index": end_index,
                "start_time": start_time,
                "end_time": end_time,
                "epoch_duration": end_time - start_time,
            }
        )
    return pd.DataFrame(epoch_list)


def calculate_dot_locations():
    return consts.dots_cm


def calculate_table_calibration(dots_pixels, dots_cm=calculate_dot_locations()):
    # OpenCV does the actual fitting. This is a non-linear fit

    if not set(dots_pixels.keys()).issubset(set(dots_cm.keys())):
        ValueError("List of pixel dots includes dots that aren't in the real dots")

    pixel_coords = []
    real_coords = []
    for k in dots_pixels.keys():
        pixel_coords.append(dots_pixels[k])
        real_coords.append(dots_cm[k])
    print(f"{pixel_coords=}")
    print(f"real_coords=")
    pixel_coords = np.array(pixel_coords)
    real_coords = np.array(real_coords)

    homography_matrix, mask = cv.findHomography(
        pixel_coords, real_coords, cv.RANSAC, ransacReprojThreshold=3.0
    )

    # And then we can use the fit to transform the dots back and see how close we
    # got to what we asked for
    transformed_real_coords = cv.perspectiveTransform(
        np.array([pixel_coords]), homography_matrix
    )[0]
    delta_dot_position = np.std(transformed_real_coords - real_coords)
    return homography_matrix, delta_dot_position


# This function is written so it can take an array or a dict
# and also so the data can be 1x2 or Nx2
def apply_table_calibration(data, homography_matrix):
    if isinstance(data, np.ndarray):
        # If input is a NumPy array
        was_array = True
        data = {"data": data}
    elif isinstance(data, dict):
        was_array = False

    result = dict()
    for k, v in data.items():
        if np.ndim(v) == 1:
            was_1d = True
            v = v.reshape([1, -1])
        else:
            was_1d = False

        source = np.array([v])
        output = cv.perspectiveTransform(source, homography_matrix)[0]

        # Find indices of NaN values in the source points
        nan_indices = np.isnan(source[0]).any(axis=1)

        # Replace NaN values in the transformed points with NaN
        output[nan_indices, :] = np.nan
        result[k] = output

        if was_1d:
            result[k] = np.squeeze(result[k])

    if was_array:
        result = result["data"]

    return result
