import os
import re
import copy

import numpy as np
import deepdish as dd

import kineticstoolkit as ktk

from markers_analysis import constants as consts
from markers_analysis import billiards_table as billiards
from markers_analysis import markers
from markers_analysis import math as my_math


def find_data_files(
    dir_path,
    basename,
    extension_list,
):
    file_list = []

    extension_string = "|".join(extension_list)
    regex = basename + r"(\d+)(.*\.({})$)".format(extension_string)

    for filename in os.listdir(dir_path):
        if any(filename.endswith(ext) for ext in extension_list):
            # Extract the number from the filename using regular expression
            match = re.search(regex, filename)

            if match:
                file_list.append(
                    (int(match.group(1)), os.sep.join([dir_path, match.group()]))
                )
            else:
                print(
                    f"The data file {filename} doesn't match pattern and is being skipped."
                )

    file_list = sorted(file_list)
    return file_list


def find_ball_data_files(
    date,
    id,
    basename = consts.table_data_basename,
    basepath = consts.data_basepath,
    path_fn = consts.table_data_path_fn,
    keep_list = None
):
    dir_path = path_fn(basepath, date, id)
    extension_list = ["csv", "h5", "hdf5"]
    file_list = find_data_files(dir_path, basename, extension_list)
    if keep_list is not None:
        file_list = list(set(file_list) & set(keep_list))
    return file_list


def find_marker_data_files(
    date,
    id,
    basename=consts.markers_data_basename,
    basepath=consts.data_basepath,
    path_fn=consts.markers_data_path_fn,
):
    dir_path = path_fn(basepath, date, id)
    extension_list = ["c3d"]
    file_list = find_data_files(dir_path, basename, extension_list)
    return file_list


# Note: for Python 2.7 compatibility, use ur"" to prefix the regex and u"" to prefix the test string and substitution


def calibrate_table(data_dict):
    transformation, delta_dot_position = billiards.calculate_table_calibration(
        data_dict["dots"]
    )

    # Now translate the other things that are important going forward.
    ts_cm = ktk.TimeSeries(
        time=data_dict["ts"].time, time_info=data_dict["ts"].time_info
    )

    ts_cm.data = billiards.apply_table_calibration(data_dict["ts"].data, transformation)
    # Add units and type information
    for k in ts_cm.data.keys():
        ts_cm.add_data_info(k, "Units", "cm", in_place=True)
        ts_cm.add_data_info(
            k, "Type", data_dict["ts"].data_info[k]["Type"], in_place=True
        )
    # Table
    table_cm = billiards.apply_table_calibration(data_dict["table"], transformation)
    data_dict_cm = {"ts": ts_cm, "table": table_cm}

    msg = f"\n  Table calibrated with an error of {delta_dot_position:.3} cm"
    print(msg)
    return data_dict_cm


def load_ball_data_file(
    filename,
    column_mapping=None,
    calibrate_table=True,
    likelihood_threshold=consts.likelihood_threshold,
):
    # Load the file into a dataframe
    data_df = billiards.load_data(
        filename,
        column_mapping=column_mapping,
        likelihood_threshold=likelihood_threshold,
    )
    msg = f"Loaded {filename}"

    # Get rid of any meaningless rows of NaNs (These shouldn't be here)
    data_df = data_df[~data_df.drop("t", axis=1).isna().all(axis=1)]

    # Turn the DataFrame into a KineticsToolbox TimeSeries
    ktk_data = (
        data_df.drop("t", axis=1, level=0)
        .T.groupby(level=0)
        .apply(lambda x: x.to_numpy().T)
        .to_dict()
    )
    ktk_time = data_df["t"].to_numpy()
    ktk_units_info = {key: {"Units": "pixels"} for key in ktk_data.keys()}

    data_ts = ktk.TimeSeries(time=ktk_time, data=ktk_data, data_info=ktk_units_info)
    for k in data_ts.data.keys():
        if k.startswith("Tbl"):
            data_ts.add_data_info(k, "Type", "Table", in_place=True)
        elif k.startswith("Dot"):
            data_ts.add_data_info(k, "Type", "Dot", in_place=True)
        elif k.endswith("ball"):
            data_ts.add_data_info(k, "Type", "Ball", in_place=True)
        else:
            data_ts.add_data_info(k, "Type", "Cue", in_place=True)

    # Get the modal points for each of the digitized Table corners
    table_names = [k for k, v in data_ts.data_info.items() if v["Type"] == "Table"]
    table = data_ts.get_subset(table_names)
    table_modes = my_math.find_modes_dict(table.data)

    # *and* get rid of the actual time series data that you got them from
    other_names = [k for k, v in data_ts.data_info.items() if v["Type"] != "Table"]
    data_ts = data_ts.get_subset(other_names)

    # Get the modal points for each of the calibrated dots
    dot_names = [k for k, v in data_ts.data_info.items() if v["Type"] == "Dot"]
    dots = data_ts.get_subset(dot_names)
    dot_modes = my_math.find_modes_dict(dots.data)

    # Remove the dot time series also from the time series
    other_names = [k for k, v in data_ts.data_info.items() if v["Type"] != "Dot"]
    data_ts = data_ts.get_subset(other_names)

    data_dict = {"table": table_modes, "dots": dot_modes, "ts": data_ts}

    # Generally, we will want to transform the units to cm
    if calibrate_table:
        msg += "\n Results returned in cm"
        data_dict_cm = calibrate_table(data_dict)
        return data_dict_cm
    else:
        msg += "\n  Results returned in pixels"
        print(msg)
        return data_dict


def load_marker_data_file(
    filename,
    interconnections
):
    data_ts = markers.read_c3d_file(filename)
    data_ts = ktk.filters.butter(data_ts, fc=consts.marker_freq_lowpass)

    frames, global_transform = markers.get_frames(data_ts)
    euler = markers.get_euler_angles(frames, global_transform)
    euler_vels = markers.get_euler_velocities(euler)

    data_dict = {
        "markers": data_ts,
        "interconnections": interconnections,
        "frames": frames,
        "euler": euler,
        "euler_vels": euler_vels, 
        "global_transform": global_transform
    }
    return data_dict



def get_calculated_data(ts, calculation_fn, calculation_name, merge_in_place=False):
    if not merge_in_place:
        ts = ts.copy()

    new_ts = calculation_fn(ts)

    names = list(ts.data.keys())
    new_names = [s + "_" + calculation_name for s in names]

    for s, new_s in zip(names, new_names):
        new_ts.rename_data(s, new_s, in_place=True)

    if merge_in_place:
        ts.merge(new_ts, in_place=True)
    else:
        return new_ts


def get_velocities(ts, ball_names=None, merge_in_place=False):
    if not ball_names:
        ball_names = [k for k, v in ts.data_info.items() if v["Type"] == "Ball"]

    balls = ts.get_subset(ball_names)

    vels = get_calculated_data(balls, billiards.get_velocities, "vel")

    if merge_in_place:
        ts.merge(vels, in_place=True, overwrite=False)
    else:
        return vels


def get_speeds(ts, ball_names=None, merge_in_place=False):
    if not ball_names:
        ball_names = [k for k, v in ts.data_info.items() if v["Type"] == "Ball"]
    vel_names = [
        k
        for k, v in ts.data_info.items()
        if (v["Type"] == "Ball Velocity" and k in ball_names)
    ]
    no_vel_names = [n for n in ball_names if n not in vel_names]

    balls = ts.get_subset(no_vel_names)
    vels = ts.get_subset(vel_names)

    new_vels = get_velocities(balls)
    vels.merge(new_vels, in_place=True)

    speeds = get_calculated_data(vels, billiards.get_speed_from_velocity, "speed")
    names = list(speeds.data.keys())
    for s in names:
        new_s = re.sub(r"vel_speed$", "speed", s)
        speeds.rename_data(s, new_s, in_place=True)

    if merge_in_place:
        ts.merge(speeds, in_place=True, overwrite=False)
    else:
        return speeds


def get_moving(ts, ball_names=None, merge_in_place=False):
    if not ball_names:
        ball_names = [k for k, v in ts.data_info.items() if v["Type"] == "Ball"]
    speed_names = [
        k
        for k, v in ts.data_info.items()
        if (v["Type"] == "Ball Speed" and k in ball_names)
    ]
    no_speed_names = [n for n in ball_names if n not in speed_names]

    balls = ts.get_subset(no_speed_names)
    speeds = ts.get_subset(speed_names)

    new_speeds = get_speeds(balls)
    speeds.merge(new_speeds, in_place=True)

    moving = get_calculated_data(speeds, billiards.is_moving, "moving")
    names = list(moving.data.keys())
    for s in names:
        new_s = re.sub(r"speed_moving$", "moving", s)
        moving.rename_data(s, new_s, in_place=True)

    if merge_in_place:
        ts.merge(moving, in_place=True, overwrite=False)
    else:
        return moving


def find_shot_start(data_dict):
    start_positions_time = consts.start_positions_time
    window_of_stillness = consts.window_of_stillness

    ts = data_dict["ts"]

    t = ts.time

    cue_ball_not_moving = ~ts.data["cue_ball_moving"]

    ball_names = [k for k, v in ts.data_info.items() if v["Type"] == "Ball"]
    ball_positions_ok = billiards.is_in_position(
        ts.data, positioned_objects_list=ball_names
    )[0]

    cue_ball = ts.data["cue_ball"]
    end_cue = ts.data["end_cue"]
    cue_position_ok = billiards.is_in_order(end_cue, cue_ball)

    start_conditions = cue_ball_not_moving & ball_positions_ok & cue_position_ok
    start_conditions_runs = billiards.find_runs(
        t, start_conditions, start_positions_time
    )

    start_times = np.maximum(0, start_conditions_runs.end_times - window_of_stillness)
    return start_times


def find_shot_end(data_dict):
    still_ball_time = consts.still_ball_time
    moving_ball_time = consts.moving_ball_time
    table_hit_window = consts.table_hit_window
    nan_run_time = consts.nan_run_time

    # If the object ball STOPS moving we're done
    ts = data_dict["ts"]
    t = ts.time
    object_ball_not_moving = ~ts.data["object_ball_moving"]
    object_ball_not_moving_runs = np.full_like(object_ball_not_moving, False)
    runs_df = billiards.find_runs(t, object_ball_not_moving, still_ball_time)
    object_ball_not_moving_runs[runs_df.start_indices] = True

    # If the target ball STARTS moving, we're done
    target_ball_moving = ts.data["target_ball_moving"]
    target_ball_moving_runs = np.full_like(target_ball_moving, False)
    runs_df = billiards.find_runs(t, target_ball_moving, moving_ball_time)
    target_ball_moving_runs[runs_df.start_indices] = True

    # If the object ball hits the table edge, we're done
    object_ball = ts.data["object_ball"]
    object_ball_vel = ts.data["object_ball_vel"]
    table = data_dict["table"]
    hit_table = billiards.hit_table_edge(
        t, object_ball, object_ball_vel, table, table_hit_window=table_hit_window
    )

    # If any ball gets covered up by a hand (and becomes NaN), we're done
    ball_names = [k for k, v in ts.data_info.items() if v["Type"] == "Ball"]
    balls = ts.get_subset(ball_names)
    nan_balls = np.array([np.isnan(v).any(axis=1) for v in balls.data.values()]).any(
        axis=0
    )
    # Keep nan runs of at least 50 ms
    nan_runs = np.full_like(nan_balls, False)
    runs_df = billiards.find_runs(t, nan_balls, nan_run_time)
    nan_runs[runs_df.start_indices] = True

    # Combine all stopping conditions with an OR
    end_conditions = object_ball_not_moving_runs | target_ball_moving_runs | hit_table

    end_times = t[end_conditions]
    return end_times


def get_shots(
    data_dict, merge_in_place=False, minimum_gap_time=consts.minimum_trial_gap
):
    start_time = find_shot_start(data_dict)
    end_time = find_shot_end(data_dict)
    shots_epochs = billiards.find_epochs(
        data_dict["ts"].time, start_time, end_time, minimum_gap_time
    )

    if merge_in_place:
        data_dict["shots"] = shots_epochs
    else:
        return shots_epochs


def is_hit(shots):
    def _is_hit(shot):
        ts = shot["ts"]
        tb = ts.data['target_ball']
        ob = ts.data['object_ball']
        d = my_math.dist(tb, ob)
        target_ball_moving = ts.get_subset("target_ball_moving")
        return (target_ball_moving.data["target_ball_moving"] | (d < consts.ball_radius)).any()

    if isinstance(shots, dict):
        return _is_hit(shots)

    return [_is_hit(shot) for shot in shots]


def count_hits(shots):
    hits = is_hit(shots)
    return np.array(hits).sum()


"""
def get_angles(shots_data):
    object_ball_angle = np.full_like(shots_data, fill_value=np.nan)

    for shotnum, shot in enumerate(shots_data):
        ts = shot["ts"]

        object_ball = ts.data["object_ball"]
        # object_ball_vel = ts.data["object_ball_vel"]
        object_ball_moving = ts.data["object_ball_moving"]
        target_ball = ts.data["target_ball"]

        if object_ball_moving.any():
            epoch_angles = billiards.get_roll_angle(
                object_ball[object_ball_moving],
                object_ball_vel[object_ball_moving],
                target_ball[object_ball_moving],
            )

            main_angle = epoch_angles.mean()
        else:
            main_angle = np.nan

        object_ball_angle[shotnum] = main_angle

    return object_ball_angle
"""


def get_angles(shots_data):
    object_ball_angle = np.full_like(shots_data, fill_value=np.nan, dtype=np.double)

    for shotnum, shot in enumerate(shots_data):
        ts = shot["ts"]

        object_ball = ts.data["object_ball"]
        object_ball_moving = ts.data["object_ball_moving"]
        target_ball = ts.data["target_ball"]
        keep = object_ball_moving & (ts.time < 1)

        if keep.any():
            main_angle = billiards.get_roll_angle(
                object_ball[keep],
                target_ball[keep],
            )
        else:
            main_angle = np.nan

        object_ball_angle[shotnum] = main_angle

    return object_ball_angle


def set_params_per_file(file_list, default_params, file_params):
    all_params = {}
    for n, filename in file_list:
        all_params[n] = {}
        for k in default_params.keys():
            if file_params and n in file_params.keys() and k in file_params[n].keys():
                all_params[n][k] = these_params[n][k]
            else:
                all_params[n][k] = default_params[k]
    return all_params


def make_file_video(data_dict, video_file):
    data_dict_for_video = copy.deepcopy(data_dict)
    end_video_time = np.min([video_time[1], data_dict_for_video["ts"].time[-1]])
    start_video_time = np.max([0, video_time[0]])

    data_dict_for_video["ts"] = data_dict_for_video["ts"].get_ts_between_times(
        start_video_time, end_video_time
    )
    billiards.create_video(
        data_dict_for_video,
        video_file,
        do_dots=True,
    )


def save_subject_to_hd5(
    subject_data,
    subject_id,
    date,
    basename=consts.table_data_basename,
    basepath=consts.data_basepath,
    path_fn=consts.table_data_path_fn,
):
    dir_path = path_fn(basepath, date, subject_id)
    filename = os.path.join(dir_path, f"{basename}.h5")

    subject_data = copy.deepcopy(subject_data)
    for data_dict in subject_data["data"]:
        data_dict["ts"] = data_dict["ts"].to_dataframe()

    print(f"Saving subject {subject_id} on {date} to {filename}")
    dd.io.save(filename, subject_data)


def load_subject_from_hd5(
    subject_id,
    date,
    basename=consts.table_data_basename,
    basepath=consts.data_basepath,
    path_fn=consts.table_data_path_fn,
):
    dir_path = path_fn(basepath, date, subject_id)
    filename = os.path.join(dir_path, f"{basename}.h5")

    print(f"Loading subject {subject_id} on {date} from {filename}")
    subject_data = dd.io.load(filename)
    for data_dict in subject_data["data"]:
        data_dict["ts"] = data_dict["ts"].from_dataframe()

    return subject_data


def load_subject_ball_files(
    subject_id,
    date,
    basename=consts.table_data_basename,
    basepath=consts.data_basepath,
    path_fn=consts.table_data_path_fn,
    column_mapping=None,
    file_params=None,
    do_video=True,
    video_directory=os.path.join(consts.results_basepath, "videos"),
    video_time=None,
    save_data=True,
    force_reload=False,
    likelihood_threshold=consts.likelihood_threshold,
):
    dir_path = path_fn(basepath, date, subject_id)
    filename = os.path.join(dir_path, f"{basename}.h5")

    if not force_reload:
        if os.path.exists(filename):
            subject_data = load_subject_from_hd5(
                subject_id, date, basename=basename, basepath=basepath, path_fn=path_fn
            )
            return subject_data

    print(f"Loading marker files for {subject_id} on {date} from {dir_path}")
    subject_data = {}

    info = {}
    info["id"] = subject_id
    info["date"] = date
    info["basename"] = basename

    subject_data["info"] = info
    subject_data["data"] = []

    default_params = {
        "column_mapping": column_mapping,
        "do_video": do_video,
        "video_time": video_time,
        "likelihood_threshold": likelihood_threshold,
    }

    file_list = find_ball_data_files(
        date=date,
        id=subject_id,
        basename=basename,
        basepath=basepath,
        path_fn=path_fn,
    )
    file_params = set_params_per_file(file_list, default_params, file_params)

    for filenum, filename in file_list:
        these_params = file_params[filenum]
        column_mapping = these_params["column_mapping"]
        do_video = these_params["do_video"]
        video_time = these_params["video_time"]

        data_dict_pixels = load_ball_data_file(
            filename, column_mapping=column_mapping, calibrate_table=False
        )

        data_dict = calibrate_table(data_dict_pixels)

        if do_video:
            video_file = f"pixels {subject_id} {filenum}.mp4"
            calibrated_video_file = f"calibrated {video_file}"
            video_file = os.path.join(video_directory, video_file)
            calibrated_video_file = os.path.join(video_directory, calibrated_video_file)
            billiards.create_video(
                data_dict_pixels,
                file_name=video_file,
                keep_time=video_time,
                do_dots=True,
            )
            billiards.create_video(
                data_dict,
                file_name=calibrated_video_file,
                keep_time=video_time,
                do_dots=True,
            )

        data_dict["filenum"] = filenum
        subject_data["data"].append(data_dict)

    if save_data:
        save_subject_to_hd5(
            subject_data,
            subject_id,
            date,
            basename=basename,
            basepath=basepath,
            path_fn=path_fn,
        )

    return subject_data
