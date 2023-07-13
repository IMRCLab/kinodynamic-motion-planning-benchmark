import numpy as np


def rot1(angle, degrees=False):
    '''
    Converts pitch angle (a rotation around the 1st body axis) to a rotation
    matrix in SO(3).

    Args:
        angle: (numpy.ndarray) pitch angle
        degrees: (bool) flag to use if the angles are in degrees,
            default = False
    Returns:
        R: (numpy.ndarray) 3x3 rotation matrix in SO(3)
    '''

    if degrees:
        angle = np.deg2rad(angle)

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rot_mat = np.identity(3)

    rot_mat[1, 1] = cos_a
    rot_mat[1, 2] = -sin_a
    rot_mat[2, 1] = sin_a
    rot_mat[2, 2] = cos_a

    return rot_mat


def rot2(angle, degrees=False):
    '''
    Converts roll angle (a rotation around the 2nd body axis) to a rotation
    matrix in SO(3).

    Args:
        angle: (numpy.ndarray) roll angle
        degrees: (bool) flag to use if the angles are in degrees,
            default = False
    Returns:
        R: (numpy.ndarray) 3x3 rotation matrix in SO(3)
    '''

    if degrees:
        angle = np.deg2rad(angle)

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rot_mat = np.identity(3)

    rot_mat[0, 0] = cos_a
    rot_mat[0, 2] = sin_a
    rot_mat[2, 0] = -sin_a
    rot_mat[2, 2] = cos_a

    return rot_mat


def rot3(angle, degrees=False):
    '''
    Converts yaw angle (a rotation around the 3rd body axis) to a rotation
    matrix in SO(3).

    Args:
        angle: (numpy.ndarray) yaw angle
        degrees: (bool) flag to use if the angles are in degrees,
            default = False
    Returns:
        R: (numpy.ndarray) 3x3 rotation matrix in SO(3)
    '''

    if degrees:
        angle = np.deg2rad(angle)

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rot_mat = np.identity(3)

    rot_mat[0, 0] = cos_a
    rot_mat[0, 1] = -sin_a
    rot_mat[1, 0] = sin_a
    rot_mat[1, 1] = cos_a

    return rot_mat


def ypr_to_R(ypr, degrees=False):
    '''
    Converts yaw, pitch, roll angles to a rotation matrix in SO(3).

    Args:
        ypr: (numpy.ndarray) 3x1 array with yaw, pitch, roll
        degrees: (bool) flag to use if the angles are in degrees,
            default = False
    Returns:
        R: (numpy.ndarray) 3x3 rotation matrix in SO(3)
    '''

    R3 = rot3(ypr[0], degrees)
    R2 = rot2(ypr[1], degrees)
    R1 = rot1(ypr[2], degrees)

    return R3.dot(R2).dot(R1)
