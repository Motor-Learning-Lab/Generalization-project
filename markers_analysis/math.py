import numpy as np

# from numba import jit

# import sklearn as skl
# from sklearn import neighbors as skln
from scipy import stats
from scipy import interpolate
from scipy.spatial import distance


# @jit
def dist(p1, p2):
    p1 = np.atleast_2d(p1)
    p2 = np.atleast_2d(p2)
    return np.sqrt(np.sum((p2 - p1) * (p2 - p1), axis=1))


# @jit
def dist2(p1, p2):
    return np.sum((p2 - p1) * (p2 - p1), axis=1)


# This math is from a stack overflow question here: https://stackoverflow.com/q/39840030
# This only works for two dimensions. There is a 3D solution later in the comments.
def dist_segment(p, s1, s2):
    return np.abs(np.cross(s2 - s1, p - s1)) / dist(s2, s1)


# Here is a derivation of the equation for the derivative of the distance in terms of the 
# derivative of the point
# \[\begin{gathered}
#   d = \frac{{\left| {\left( {{s_2} - {s_1}} \right) \times \left( {p - {s_1}} \right)} \right|}}{{\left\| {{s_2} - {s_1}} \right\|}} \\ 
#   \frac{{dd}}{{dt}} = \frac{d}{{dt}}\frac{{\left| {\left( {{s_2} - {s_1}} \right) \times \left( {p - {s_1}} \right)} \right|}}{{\left\| {{s_2} - {s_1}} \right\|}} \\ 
#    = \frac{{\left( {\left( {{s_2} - {s_1}} \right) \times \left( {p - {s_1}} \right)} \right) \cdot \left( {\frac{d}{{dt}}\left( {{s_2} - {s_1}} \right) \times \left( {p - {s_1}} \right)} \right)}}{{\left\| {\left( {{s_2} - {s_1}} \right) \times \left( {p - {s_1}} \right)} \right\|\left\| {{s_2} - {s_1}} \right\|}} \\ 
#    = \frac{{\left( {\left( {{s_2} - {s_1}} \right) \times \left( {p - {s_1}} \right)} \right) \cdot \left( {\left( {{s_2} - {s_1}} \right) \times \left( {v - {s_1}} \right)} \right)}}{{\left\| {\left( {{s_2} - {s_1}} \right) \times \left( {p - {s_1}} \right)} \right\|\left\| {{s_2} - {s_1}} \right\|}} \\ 
# \end{gathered} \]
# But since we are in two dimensions, we can just multiply the cross products instead of dot producting them.
def d_dist_segment(p, v, s1, s2):
    s = s2 - s1
    num = np.cross(s, p-s1)*np.cross(s, v-s1)
    den = np.abs( np.cross(s, p-s1))*np.linalg.norm(s)
    return num / den


# ## Get starting position for each ball
# Because I don't have data from before and also because it's more robust: the starting
# for each ball is calculated as the position where it spends the most time. This makes
# sense because shots are short and most of the time is actually spent waiting for the
# next shot.
# Finding the mode of an estimated distribution is apparently not a very solved problem
# (or maybe it is and that's all of machine learning).
# Generally, people estimate the distribution with a Gaussian kde and then use a find
# grid to approximate the maximum. I replaced the grid with resampling because that
# likely to be more accurate. I'm still a bit troubled that the bandwidth of the KDE
# isn't doing a good job of getting an approximation of the true KDE. The figure
# above is not very comforting. It may be worth looking into this, but the results
# seems pretty convincing and the accuracy we need is not large.
# @jit
def find_mode_array(arr):
    if arr.ndim == 1:
        # For 1D vectors, reshape to a column vector
        arr = arr.reshape(-1, 1)

    arr = arr[~np.isnan(arr).any(axis=1)]

    if arr.shape[0] > 1:
        # If there is more than one value, get non-parametric fit and take a real mode
        kde = stats.gaussian_kde(arr.T)

        # Generate a resampling of points
        resamp = kde.resample(20000)

        # Compute the log-density at each point on the grid
        logpdf = kde.logpdf(resamp)

        # Find the mode (peak) of the estimated density
        mode_indices = np.unravel_index(np.argmax(logpdf), logpdf.shape)
        mode_indices = mode_indices[0]
        mode = resamp[:, mode_indices]
    elif arr.shape[0] > 0:
        # If there is only one value, it is the mode
        mode = arr[0, :]
    else:
        # If there isn't even one, then there is no mode
        mode = np.full((1, arr.shape[1]), np.nan)

    return mode.flatten()


def find_modes_dict(data_dict, label_list=None):
    if label_list:
        data_dict = {k: v for k, v in data_dict.items() if k in label_list}

    modes_dict = dict()
    for k, v in data_dict.items():
        modes_dict[k] = find_mode_array(v)

    return modes_dict
