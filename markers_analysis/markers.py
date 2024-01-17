# read_markers.py
#
import kineticstoolkit.lab as ktk


def read_c3d_file(data_file):
    markers = ktk.read_c3d(data_file, convert_point_unit=True)["Points"]

    for average_marker, mark1, mark2 in zip(
        ["L_ELB", "R_ELB", "L_W", "R_W"],
        ["L_ELB_L", "R_ELB_L", "L_W_L", "R_W_L"],
        ["L_ELB_M", "R_ELB_M", "L_W_M", "R_W_M"],
    ):
        markers.data[average_marker] = (markers.data[mark1] + markers.data[mark2]) / 2

    return markers


def get_interconnections():
    interconnections = dict()  # Will contain all segment definitions

    interconnections["Back"] = {
        "Color": [0, 0.5, 1],  # In RGB format (here, greenish blue)
        "Links": [  # List of lines that span lists of markers
            ["R_SAE", "L5/S1", "R_GR"],
            ["L_SAE", "L5/S1", "L_GR"],
            ["R_SAE", "L_SAE"],
            ["R_GR", "L_GR"],
        ],
    }

    interconnections["LUpperLimb"] = {
        "Color": [0, 0.5, 1],
        "Links": [
            ["L_SAE", "L_ELB", "L_W", "L_H_M"],
            ["L_W", "L_H_L", "L_H_M"],
        ],
    }

    interconnections["RUpperLimb"] = {
        "Color": [1, 0.5, 0],
        "Links": [
            ["R_SAE", "R_ELB", "R_W", "R_H_M"],
            ["R_W", "R_H_L", "R_H_M"],
        ],
    }

    interconnections["Cue"] = {
        "Color": [0.5, 1, 0.5],
        "Links": [
            ["FQ", "BQ"],
        ],
    }
    return interconnections


def get_frames(markers):
    frames = ktk.TimeSeries(time=markers.time)

    # ## Global
    # The global coordinate system was recorded with:
    #            z up, y forward and x to the right.
    # The joint coordinates are in y up, x forward, and z to the right.
    # Let's create a frame that captures this.
    global_transform = ktk.geometry.create_transforms(
        seq="xy", angles=[[-90, -90]], degrees=True
    )

    # ## Pelvis
    pelvis_origin = (markers.data["R_GR"] + markers.data["L_GR"]) / 2
    pelvis_z = markers.data["L_GR"] - markers.data["R_GR"]
    pelvis_yz = markers.data["L5/S1"] - pelvis_origin

    frames.data["Pelvis"] = ktk.geometry.create_frames(
        origin=pelvis_origin, z=pelvis_z, yz=pelvis_yz
    )

    # ## Upper back
    thorax_origin = (markers.data["L_SAE"] + markers.data["R_SAE"]) / 2
    thorax_z = markers.data["L_SAE"] - markers.data["R_SAE"]
    thorax_yz = thorax_origin - markers.data["L5/S1"]

    frames.data["Thorax"] = ktk.geometry.create_frames(
        origin=thorax_origin, z=thorax_z, yz=thorax_yz
    )

    # ## Upper arms
    # ### Right upper arm
    r_arm_origin = markers.data["R_SAE"]
    r_arm_y = markers.data["R_SAE"] - markers.data["R_ELB"]
    r_arm_yz = markers.data["R_ELB_L"] - markers.data["R_ELB_M"]

    frames.data["R_Arm"] = ktk.geometry.create_frames(
        origin=r_arm_origin, y=r_arm_y, yz=r_arm_yz
    )

    # ### Left upper arm
    l_arm_origin = markers.data["L_SAE"]
    l_arm_y = markers.data["L_SAE"] - markers.data["L_ELB"]
    l_arm_yz = markers.data["L_ELB_L"] - markers.data["L_ELB_M"]

    frames.data["L_Arm"] = ktk.geometry.create_frames(
        origin=l_arm_origin, y=l_arm_y, yz=l_arm_yz
    )

    # ## Forearms
    # ### Right forearm
    r_forearm_origin = markers.data["R_W"]
    r_forearm_y = markers.data["R_ELB"] - markers.data["R_W"]
    r_forearm_yz = markers.data["R_W_M"] - markers.data["R_W_L"]

    frames.data["R_Forearm"] = ktk.geometry.create_frames(
        origin=r_forearm_origin, y=r_forearm_y, yz=r_forearm_yz
    )

    # ### Left forearm
    l_forearm_origin = markers.data["L_W"]
    l_forearm_y = markers.data["L_ELB"] - markers.data["L_W"]
    l_forearm_yz = markers.data["L_W_M"] - markers.data["L_W_L"]

    frames.data["L_Forearm"] = ktk.geometry.create_frames(
        origin=l_forearm_origin, y=l_forearm_y, yz=l_forearm_yz
    )

    # ## Hands
    # ### Right hand
    r_hand_origin = markers.data["R_W_L"]
    r_hand_z = markers.data["R_W_L"] - markers.data["R_W_M"]
    r_hand_yz = (markers.data["R_H_M"] + markers.data["R_H_L"]) / 2 - markers.data[
        "R_W"
    ]

    frames.data["R_Hand"] = ktk.geometry.create_frames(
        origin=r_hand_origin, z=r_hand_z, yz=r_hand_yz
    )

    # ### Left hand
    l_hand_origin = markers.data["L_W_L"]
    l_hand_z = markers.data["L_W_M"] - markers.data["L_W_L"]
    l_hand_yz = markers.data["L_W"] - (markers.data["L_H_M"] + markers.data["L_H_L"]) / 2

    frames.data["L_Hand"] = ktk.geometry.create_frames(
        origin=l_hand_origin, z=l_hand_z, yz=l_hand_yz
    )

    return frames, global_transform


def get_homogeneous_angle(frames, global_transform):
    homogeneous = ktk.TimeSeries(time=frames.time)

    homogeneous.data["pelvis"] = ktk.geometry.get_local_coordinates(
        frames.data["Pelvis"], global_transform
    )
    homogeneous.data["pelvis_to_thorax"] = ktk.geometry.get_local_coordinates(
        frames.data["Thorax"], frames.data["Pelvis"]
    )
    homogeneous.data["thorax_to_r_arm"] = ktk.geometry.get_local_coordinates(
        frames.data["R_Arm"], frames.data["Thorax"]
    )
    homogeneous.data["thorax_to_l_arm"] = ktk.geometry.get_local_coordinates(
        frames.data["L_Arm"], frames.data["Thorax"]
    )
    homogeneous.data["R_arm_to_forearm"] = ktk.geometry.get_local_coordinates(
        frames.data["R_Forearm"], frames.data["R_Arm"]
    )
    homogeneous.data["L_arm_to_forearm"] = ktk.geometry.get_local_coordinates(
        frames.data["L_Forearm"], frames.data["L_Arm"]
    )
    homogeneous.data["R_forearm_to_hand"] = ktk.geometry.get_local_coordinates(
        frames.data["R_Hand"], frames.data["R_Forearm"]
    )
    homogeneous.data["L_forearm_to_hand"] = ktk.geometry.get_local_coordinates(
        frames.data["L_Hand"], frames.data["L_Forearm"]
    )

    return homogeneous


def get_euler_angles(frames, global_transform):
    homogeneous = get_homogeneous_angle(frames, global_transform)

    euler = ktk.TimeSeries(time=frames.time)

    euler.data["Pelvis"] = ktk.geometry.get_angles(
        homogeneous.data["pelvis"], "YZX", degrees=True
    )
    euler.data["Thorax"] = ktk.geometry.get_angles(
        homogeneous.data["pelvis_to_thorax"], "YZX", degrees=True
    )
    euler.data["R_Arm"] = ktk.geometry.get_angles(
        homogeneous.data["thorax_to_r_arm"], "YXY", degrees=True
    )
    euler.data["L_Arm"] = ktk.geometry.get_angles(
        homogeneous.data["thorax_to_l_arm"], "YXY", degrees=True
    )
    euler.data["R_Forearm"] = ktk.geometry.get_angles(
        homogeneous.data["R_arm_to_forearm"], "ZYX", degrees=True
    )
    euler.data["L_Forearm"] = ktk.geometry.get_angles(
        homogeneous.data["L_arm_to_forearm"], "ZYX", degrees=True
    )
    euler.data["R_Hand"] = ktk.geometry.get_angles(
        homogeneous.data["R_forearm_to_hand"], "XZY", degrees=True
    )
    euler.data["L_Hand"] = ktk.geometry.get_angles(
        homogeneous.data["L_forearm_to_hand"], "XZY", degrees=True
    )

    return euler


# ## Get the veocities
# For the joints with Cardan angles, the change
# is just the derivative of the angle itself.
# For instance, for pelvis, the three angles are facing (called rotation clinically),
# body tilt (called tilt), and then what I would call the angle
# (one side being higher than the other, called obliquity in the literature).
# In that case, we can just take their derivatives.
# These joints have Cardan angles: Pelvis, Thorax, and the forearms
# For the joints with Euler angles, things are similar but a little different.
# For example, for the shoulder is the elevation plane, the elevation,
# and then humeral rotation. So, the change in the elevation plane
# is the still the derivative of the first variable,
# the change in elevation is the derivative of the second,
# but the derivative of the humeral rotation is different.
# That's because the change in Y that moves the elevation plane
# has to be undone by increased humeral rotation.
# Thus, the change in the actual humeral rotation
# is the derivative of the sum of these two.
# These joint have Euler angles: Shoulder


def get_euler_velocities(euler):
    euler_vels = ktk.filters.deriv(euler)
    euler_vels.resample(euler.time, kind='cubic', in_place=True)

    euler_vels.data["R_Arm"][2] = (
        euler_vels.data["R_Arm"][0] + euler_vels.data["R_Arm"][2]
    )
    euler_vels.data["L_Arm"][2] = (
        euler_vels.data["L_Arm"][0] + euler_vels.data["L_Arm"][2]
    )

    return euler_vels
