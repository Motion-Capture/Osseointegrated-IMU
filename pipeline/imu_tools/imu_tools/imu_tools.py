from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import Akima1DInterpolator
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from imu_tools.transform import *
from vqf import VQF, offlineVQF

def error_metric(reference, test, axis=0):
    '''
    Calculate the error metrics between reference and test arrays
    '''
    # MAE RMS MAX_ERR ROM REL_ERR CORR
    errors = {}
    errors['mae'] = np.mean(np.abs(reference - test), axis=axis)
    errors['rms'] = np.sqrt(np.mean((reference - test)**2, axis=axis))
    errors['max_err'] = np.max(np.abs(reference - test), axis=axis)
    errors['rom_ref'] = np.max(reference, axis=axis) - np.min(reference, axis=axis)
    errors['rom_test'] = np.max(test, axis=axis) - np.min(test, axis=axis)
    errors['rel_err'] = errors['rms'] / errors['rom_ref']
    errors['ame'] = np.abs(np.mean(reference - test, axis=axis))
    errors['me'] = np.mean(reference - test, axis=axis)
    errors['std'] = np.std(reference - test, axis=axis)
    # correlation
    ref = reference.reshape(reference.shape[0], -1, order='F')
    tst = test.reshape(test.shape[0], -1, order='F')
    corr = np.zeros(ref.shape[1])
    for i in range(ref.shape[1]):
        corr[i] = np.corrcoef(ref[:,i], tst[:,i])[0,1]
    corr = corr.reshape(reference.shape[1:], order='F')
    errors['corr'] = corr
    return errors

def error_metric_from_array(reference, test, joints):
    assert reference.shape == test.shape
    assert reference.shape[1] == len(joints)
    data = {'joint':[]}
    for i, joint in enumerate(joints):
        joint_cam = reference[:,i]
        joint_imu = test[:,i]
        errors = error_metric(joint_cam, joint_imu)
        data['joint'].append(joint)
        for key in errors:
            if key in data:
                data[key].append(errors[key])
            else:
                data[key] = [errors[key]]
    return pd.DataFrame(data)


def error_metric_from_rtable(reference, test, joints):
    data  = {'joint':[]}
    for joint in joints:
        if joint == 'pelvis':
            euler_seq = 'ZYX'
            arr_cam = reference['pelvis_imu'].as_euler(euler_seq, degrees=True)
            arr_imu = test['pelvis_imu'].as_euler(euler_seq, degrees=True)
        else:
            euler_seq = 'XYZ'
            seg_proximal = joints[joint][0]
            seg_distal = joints[joint][1]
            joint_cam = reference[seg_proximal].inv() * reference[seg_distal]
            joint_imu = test[seg_proximal].inv() * test[seg_distal]
            arr_cam = joint_cam.as_euler(euler_seq, degrees=True)
            arr_imu = joint_imu.as_euler(euler_seq, degrees=True)
        for i, ax in enumerate(euler_seq):
            errors = error_metric(arr_cam[:,i], arr_imu[:,i])
            data['joint'].append(joint + '_' + ax)
            for key in errors:
                if key in data:
                    data[key].append(errors[key])
                else:
                    data[key] = [errors[key]]
    return pd.DataFrame(data)

def bootstrap_CI(method, reference, test, n=1000, *args, **kwargs):
    '''
    method: function that calculates the error metric
    reference and test: variables to bootstrap (F x G: F frames, G gait cycles)
    n: number of bootstrap samples 
    '''
    # sample statistics
    sample_stat = method(reference, test, *args, **kwargs)
    # bootstrap
    F = reference.shape[0]
    G = reference.shape[1]
    idx = np.random.randint(0, F, (n, F))
    bootstrapped_reference = reference[idx,:] # n x F x G
    bootstrapped_test = test[idx,:]
    errors = np.zeros(n)
    for i in range(n):
        errors[i] = method(bootstrapped_reference[i,:,:], bootstrapped_test[i,:,:], *args, **kwargs)
    # confidence intervals
    errors_CI = np.quantile(errors - sample_stat, [0.975, 0.025])
    return sample_stat, sample_stat - errors_CI


def CMC(cam_frame, imu_frame):
    '''
    Calculates the CMC between the cam_frame and imu_frame
    cam_frame and imu_frame: F * G frames where F is the number of frame in the gait cycle and
    G is the number of gait cycles
    '''
    G = cam_frame.shape[1]
    F = cam_frame.shape[0]
    P = 2
    CMC_num = 0
    CMC_den = 0
    for g in range(cam_frame.shape[1]):
        g_array = np.vstack((cam_frame[:,g], imu_frame[:,g]))
        mean_g = np.mean(g_array, axis=None)
        mean_gf = np.mean(g_array, axis=0)
        g_num = np.sum((g_array - mean_gf)**2, axis=None) / (G*F*(P-1))
        g_den = np.sum((g_array - mean_g)**2, axis=None) / (G*(P*F-1))
        CMC_num += g_num
        CMC_den += g_den
    CMC = np.sqrt(1 - CMC_num / CMC_den)
    return CMC

def ellipsoid_fit(s):
    ''' Estimate ellipsoid parameters from a set of points.

        Parameters
        ----------
        s : array_like
            The samples (3,N) where N=number of samples.

        Returns
        -------
        M, n, d : array_like, array_like, float
            The ellipsoid parameters M, n, d.

        References
        ----------
        .. [1] Qingde Li; Griffiths, J.G., "Least squares ellipsoid specific
            fitting," in Geometric Modeling and Processing, 2004.
            Proceedings, vol., no., pp.335-340, 2004
    '''
    # D (samples)
    D = np.array([s[0]**2., s[1]**2., s[2]**2.,
                    2.*s[1]*s[2], 2.*s[0]*s[2], 2.*s[0]*s[1],
                    2.*s[0], 2.*s[1], 2.*s[2], np.ones_like(s[0])])

    # S, S_11, S_12, S_21, S_22 (eq. 11)
    S = np.dot(D, D.T)
    S_11 = S[:6,:6]
    S_12 = S[:6,6:]
    S_21 = S[6:,:6]
    S_22 = S[6:,6:]

    # C (Eq. 8, k=4)
    C = np.array([[-1,  1,  1,  0,  0,  0],
                    [ 1, -1,  1,  0,  0,  0],
                    [ 1,  1, -1,  0,  0,  0],
                    [ 0,  0,  0, -4,  0,  0],
                    [ 0,  0,  0,  0, -4,  0],
                    [ 0,  0,  0,  0,  0, -4]])

    # v_1 (eq. 15, solution)
    E = np.dot(np.linalg.inv(C), S_11 - np.dot(S_12, np.dot(np.linalg.inv(S_22), S_21)))

    E_w, E_v = np.linalg.eig(E)

    v_1 = E_v[:, np.argmax(E_w)]
    if v_1[0] < 0: v_1 = -v_1

    # v_2 (eq. 13, solution)
    v_2 = np.dot(np.dot(-np.linalg.inv(S_22), S_21), v_1)

    # quadratic-form parameters, parameters h and f swapped as per correction by Roger R on Teslabs page
    M = np.array([[v_1[0], v_1[5], v_1[4]],
                    [v_1[5], v_1[1], v_1[3]],
                    [v_1[4], v_1[3], v_1[2]]])
    n = np.array([[v_2[0]],
                    [v_2[1]],
                    [v_2[2]]])
    d = v_2[3]
    M_1 = np.linalg.inv(M)
    b = -np.dot(M_1, n)
    A_1 = np.real(1 / np.sqrt(np.dot(n.T, np.dot(M_1, n)) - d) * scipy.linalg.sqrtm(M))
    return b, A_1

def calib_mag(mag_data):
    '''
    Calibrate the magnetometer data using the ellipsoid fit method.
    mag_data -> M x 3 array of magnetometer data
    '''
    # fit ellipsoid
    b, Ainv = ellipsoid_fit(mag_data.T)
    # normalize
    mag_data_norm = np.dot(Ainv, (mag_data.T - b)).T
    return mag_data_norm

def XsensVQF(folder_path, file_prefix, Ts, is9D=False, calibrate_mag=False, params={}):
    # load data
    acc_path = Path(folder_path, file_prefix + '_linearAccelerations.sto')
    gyro_path = Path(folder_path, file_prefix + '_angularVelocities.sto')
    acc_data, t = get_data_from_sto(acc_path)
    gyro_data, _ = get_data_from_sto(gyro_path)
    if is9D:
        mag_path = Path(folder_path, file_prefix + '_magneticNorthHeadings.sto')
        mag_data, _ = get_data_from_sto(mag_path)
        if calibrate_mag:
            for seg in mag_data:
                if seg != 'time':
                    c = calib_mag(mag_data[seg])
                    mag_data[seg] = np.ascontiguousarray(c) # make it contiguous
    else:
        mag_data = None
    orientation = VQF_generic(acc_data, gyro_data, mag_data, Ts, params)
    return t, orientation

def VQF_generic(acc, vel, mag, Ts, params={}):
    orientation = {}
    for seg in acc:
        if mag:
            seg_orientation = offlineVQF(vel[seg], acc[seg], mag[seg], Ts=Ts, params=params)
            seg_orientation = R.from_quat(np.roll(seg_orientation['quat9D'], -1, axis=1))
        else:
            seg_orientation = offlineVQF(vel[seg], acc[seg], None, Ts=Ts, params=params)
            seg_orientation = R.from_quat(np.roll(seg_orientation['quat6D'], -1, axis=1))
        orientation[seg] = seg_orientation
    return orientation


def fetch_header(file):
    header = []
    with open(file) as f:
        for line in f:
            header.append(line.strip())
            if line.strip() == 'endheader':
                break
    return header


def get_data_from_mot(path):
    header_lines = fetch_header(path)
    header_end = len(header_lines)
    data_pd = pd.read_csv(path, header=header_end, sep='\t',
                          skipinitialspace=True, skip_blank_lines=False)
    data_header = data_pd.columns
    meta = {'header':header_lines, 'cols':data_header}
    return(meta, data_pd.to_numpy())


def get_data_from_cam(path, run_idx, FS_CAM=240,
                      interpolate=False, remove_outliers=False,
                      interpolate_method='akima', fix_first_frame=False):
    '''
    path: path to the cam data file
    run_idx: index of the run
    FS_CAM: sampling frequency of the cam data
    interpolate: whether to interpolate the data
    interpolate_method: method of interpolation

    return: RotationTable object representing segment rotations.
    If interpolate is True, the missing data will be interpolated using the specified method.
    If interpolate is False, the missing data will be dropped. !TODO! should we replace missing data with nan?
    '''

    data = pd.read_csv(path, sep='\t',
                       skiprows=5, header=None,
                       index_col=0, dtype=np.float64) # !TODO! skiprows=5 is hard coded
    if fix_first_frame:
         # fix the first frame for calibration
        data.iloc[0:FS_CAM//2,:] = np.mean(data.iloc[FS_CAM//4:FS_CAM//2,:], axis=0)

    data.index = pd.to_numeric(((data.index-1) / FS_CAM)) # convert to seconds

    joints = ['pelvis_l', 'hip_l', 'knee_l', 'ankle_l',
            'pelvis_r', 'hip_r', 'knee_r', 'ankle_r']
    cols = [j+'_'+c for j in joints for c in ['x', 'y', 'z']]

    data_run = data.iloc[:,(run_idx-1)*24:run_idx*24]
    if remove_outliers:
        isoutlier = (np.abs(scipy.stats.zscore(data_run, nan_policy='omit')) > 3)
        data_run = data_run.where(~isoutlier)
    data_run.columns = cols
    data_run_intpd = data_run.interpolate(method=interpolate_method, axis=0, limit_area='inside')
    outside_na_idx = data_run_intpd.isna().any(axis=1)
    data_run = data_run[~outside_na_idx]
    data_run_intpd = data_run_intpd[~outside_na_idx]
    inside_na_idx = data_run.index[data_run.isna().any(axis=1)]
    data_run.drop(inside_na_idx, axis=0, inplace=True)

    joint_index = joints.index('knee_r')
    plt.plot(data_run[joints[joint_index]+'_x'])
    if interpolate:
        plt.plot(data_run_intpd[joints[joint_index]+'_x'])
        plt.plot(data_run_intpd[joints[joint_index]+'_x'][inside_na_idx],
                'o', markersize=1)
        plt.legend(['raw', 'interpolated', 'interpolated data points'])
    
    if interpolate:
        data_run = data_run_intpd.copy()

    R_pelvis = R.from_euler('ZYX', pd.concat([data_run['pelvis_r_z'],
                                            -data_run['pelvis_r_y'],
                                            -data_run['pelvis_r_x']], axis=1), degrees=True)

    R_hip_r = R.from_euler('XYZ', pd.concat([data_run['hip_r_x'],
                                            data_run['hip_r_y'],
                                            data_run['hip_r_z']], axis=1), degrees=True)

    R_hip_l = R.from_euler('XYZ', pd.concat([data_run['hip_l_x'],
                                            -data_run['hip_l_y'],
                                            -data_run['hip_l_z']], axis=1), degrees=True)

    R_knee_r = R.from_euler('XYZ', pd.concat([-data_run['knee_r_x'],
                                            data_run['knee_r_y'],
                                            data_run['knee_r_z']], axis=1), degrees=True)

    R_knee_l = R.from_euler('XYZ', pd.concat([-data_run['knee_l_x'],
                                            -data_run['knee_l_y'],
                                            -data_run['knee_l_z']], axis=1), degrees=True)

    R_ankle_r = R.from_euler('XYZ', pd.concat([data_run['ankle_r_x'],
                                            data_run['ankle_r_y'],
                                            data_run['ankle_r_z']], axis=1), degrees=True)

    R_ankle_l = R.from_euler('XYZ', pd.concat([data_run['ankle_l_x'],
                                            -data_run['ankle_l_y'],
                                            -data_run['ankle_l_z']], axis=1), degrees=True)
    Rs_table = {'pelvis_imu': R_pelvis,
                'femur_r_imu': R_pelvis * R_hip_r,
                'femur_l_imu': R_pelvis * R_hip_l,
                'tibia_r_imu': R_pelvis * R_hip_r * R_knee_r,
                'tibia_l_imu': R_pelvis * R_hip_l * R_knee_l,
                'calcn_r_imu': R_pelvis * R_hip_r * R_knee_r * R_ankle_r,
                'calcn_l_imu': R_pelvis * R_hip_l * R_knee_l * R_ankle_l}

    Rs_cam = RotationTable(table=Rs_table, t=data_run.index)

    return Rs_cam


def get_data_from_sto(path):
    '''
    Returns a dictionary of data and timestamps from a .sto file.
    The first column is time, the rest are data.
    '''
    data = {}
    header_lines = fetch_header(path)
    header_end = len(header_lines)
    data_pd = pd.read_csv(path, header=header_end, sep='\t',
                            skipinitialspace=True, skip_blank_lines=False)
    segs = data_pd.columns[1:] # first column is time
    t = data_pd['time'].to_numpy()
    for seg in segs:
        data[seg] = np.array([np.fromstring(s, sep=',') for s in data_pd[seg].values])
    return data, t


def change_base_to_seg(rots, body=1, base_seg=0):
    '''
    Changes data's base frame to a specific body segment whose
    index is defined by base_seg. 0 usually means pelvis
    body=0/1 -> base/body frame;
    body -> find R so that base*R = seg
    base -> find R so that R*base = seg
    '''
    cal = []
    for r in rots:
        sample_cal = []
        base = r[base_seg]
        for seg in r:
            if body == 1:
                r_cal = base.inv() * seg # body frame
            else:
                r_cal = seg * base.inv() # base frame
            sample_cal.append(r_cal)
        cal.append(sample_cal)
    return cal


def change_base_to_sample(rots, body=1, base_sample=0):
    '''
    Changes data's base frame to a specific sample whose
    index is defined by base_sample. body=0/1 -> base/body frame;
    '''
    cal = []
    base = rots[base_sample]
    for r in rots:
        sample_cal = []
        for i, seg in enumerate(r):
            if body == 1:
                r_cal = base[i].inv() * seg # body frame
            else:
                r_cal = seg * base[i].inv() # base frame
            sample_cal.append(r_cal)
        cal.append(sample_cal)
    return cal


def rot_from_a_to_b(a, b):
    '''
    Returns the rotation required to go from vector a to vector b.
    '''
    r = np.cross(a, b)
    r_ = r / np.linalg.norm(r)
    theta = np.arctan2(np.linalg.norm(r), np.dot(a, b))
    rot_vec = theta * r_
    rot = R.from_rotvec(rot_vec)
    return rot

def resample(rots, num):
    '''
    rots -> list of lists; rotations table
    num -> number of samples in the resampled signal
    '''
    np_rots = flatten(rots, R.as_matrix)

    # do the resampling
    x = np.arange(len(rots))
    new_x = np.linspace(0, len(rots), num, endpoint=False)
    akima_interp = Akima1DInterpolator(x, np_rots, axis=0)
    np_rots_rs = akima_interp(new_x) # extrapolate?

    # rearrange into rotation table
    rots_rs = unflatten(np_rots_rs, R.from_matrix)

    # return
    return rots_rs

def parse_array(arr, markers, rs_len):
    '''
    arr -> An array object (not a table) (M*S)
    markers -> N*2 array where N is the number of slices
    each row contains the beginning and the end of a slice
    rs_len -> length of the resampled slice
    return -> Array with a shape of (N*rs_len) * S
    '''
    if arr.size == 0:
        return arr
    if arr.ndim == 1:
        arr = np.expand_dims(arr, axis=1)
    slices = []
    for marker in markers:
        slice = arr[marker[0]:marker[1]+1,:]
        slice_len = len(slice)
        akima_interp = Akima1DInterpolator(np.arange(slice_len), slice, axis=0)
        slice_rs = akima_interp(np.linspace(0, slice_len-1, rs_len)) # extrapolate?
        slices.append(slice_rs)
    parsed = np.vstack(slices)
    return parsed

def save_steps(X, X_hdr, path_steps, step_len, save=True):
    '''
    X -> data matrix
    X -> data header
    path_steps -> path to steps file
    step_len -> desired number of samples in each step
    Saves the matrix to a file with the same name as path_steps but in csv format.
    Returns the steps matrix.
    '''
    strikes = np.loadtxt(path_steps, delimiter=',', dtype=int) - 1 # -1 for matlab to python
    steps = parse_array(X, strikes, step_len)
    # save step matrix
    hdr = f'StepsCount={len(strikes)}\n' + f'StepLength={step_len}\n' + ','.join(X_hdr)
    if save:
        np.savetxt(path_steps.with_suffix('.csv'), steps, fmt='%.8f', delimiter=',',
                comments='', header=hdr)
    return steps

### AX = YB solvers and helpers ###
# solve AX = YB problem given A and B
# methods 1 and 2 are mathematically equivalent
# methods 3 and 4 are mathematically equivalent
def solveAXYB1(A, B):
    '''
    A, B -> stacked rotations of length N
    Returns X, Y as rotation objects
    '''
    N = len(A)
    M = np.zeros((4,4))
    for i in range(N):
        QA = quaternion_from_matrix(A[i].as_matrix())
        QB = quaternion_from_matrix(B[i].as_matrix())

        RA = matrix_representation(QA, PLUS)
        RB = matrix_representation(QB, MINUS)

        M += RA @ RB.T
    
    U, S, Vt = np.linalg.svd(M)
    X = quaternion_matrix(Vt[0,:])
    Y = quaternion_matrix(U[:,0])
    X = R.from_matrix(X[0:3,0:3])
    Y = R.from_matrix(Y[0:3,0:3])
    return X, Y

def solveAXYB2(A, B):
    N = len(A)
    M = np.zeros( (8,8) )

    for i in range(N):
        QA = quaternion_from_matrix(A[i].as_matrix())
        QB = quaternion_from_matrix(B[i].as_matrix())

        RA = matrix_representation(QA, PLUS)
        RB = matrix_representation(QB, MINUS)

        C = np.zeros( (4,8) )
        C[0:4,0:4] = RA
        C[0:4,4:8] = RB
        M += C.T.dot(C)

    [u,s,vt] = np.linalg.svd(M)

    X = quaternion_matrix(vt[-1,0:4])
    Y = quaternion_matrix(vt[-1,4:8])
    X = R.from_matrix(X[0:3,0:3])
    Y = R.from_matrix(Y[0:3,0:3])

    return (X, Y)

def solveAXYB3(A, B):
    N = len(A)
    I = np.identity(9)
    M = np.zeros( (18,18) )

    for i in range(N):
        RA = A[i].as_matrix()
        RB = B[i].as_matrix()

        C = np.zeros( (9,18) )
        # C[0:9,   0:9 ] = linalg.kron(RA, I)
        # C[0:9,   9:18] = -linalg.kron(-I, RB.T)
        C[0:9,   0:9 ] = np.kron(RA, RB)
        C[0:9,   9:18] = -I

        M += C.T.dot(C)

    [u,s,vt] = np.linalg.svd(M)

    X = orthonormalize_rotation(vector_matrix(vt[-1,0:9]))
    X = R.from_matrix(X[0:3,0:3])

    Y = orthonormalize_rotation(vector_matrix(vt[-1,9:18]))
    Y = R.from_matrix(Y[0:3,0:3])

    return (X, Y)

def solveAXYB4(A, B):
    N = len(A)
    M = np.zeros( (9,9) )

    for i in range(N):
        RA = A[i].as_matrix()
        RB = B[i].as_matrix()

        M += np.kron(RA, RB)

    [u,s,vt] = np.linalg.svd(M)
    X = orthonormalize_rotation(vector_matrix(vt[0,0:9]))
    X = R.from_matrix(X[0:3,0:3])

    Y = orthonormalize_rotation(vector_matrix(u[0:9,0]))
    Y = R.from_matrix(Y[0:3,0:3])

    return (X, Y)

# plot rotation matrix
def plot_rotation(rot, p_ax=None):
    R_matrix = rot.as_matrix()
    if p_ax is None:
        ax = plt.figure().add_subplot(projection='3d')
    else:
        ax = p_ax

    # plot base frame
    ax.quiver(0, 0, 0, 1, 0, 0, color='r')
    ax.quiver(0, 0, 0, 0, 1, 0, color='g')
    ax.quiver(0, 0, 0, 0, 0, 1, color='b')

    # Plot rotated frame
    ax.quiver(0, 0, 0, R_matrix[0,0], R_matrix[1,0], R_matrix[2,0], color='r', linestyle='--')
    ax.quiver(0, 0, 0, R_matrix[0,1], R_matrix[1,1], R_matrix[2,1], color='g', linestyle='--')
    ax.quiver(0, 0, 0, R_matrix[0,2], R_matrix[1,2], R_matrix[2,2], color='b', linestyle='--')

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if p_ax is None:
        plt.show()

# animate rotation matrix
# def animate_rotation(rots):
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     def update(i):
#         ax.clear()
#         plot_rotation(rots[i], ax)
#     anim = animation.FuncAnimation(fig, update, frames=len(rots), interval=10, blit=False)
#     anim.save('rotation_animation.gif', writer='imagemagick')


class RotationTable:
    def __init__(self, path=None, table={}, t=[], meta={}):
        if path is not None:
            self.read_from_STO(path)
        else:
            self.meta = meta
            self.rtable = table
            self.t = np.array(t)

    def copy(self, clean=False):
        if not clean:
            r_new = RotationTable(table=self.rtable.copy(),
                                  meta=self.meta.copy(),
                                  t=self.t.copy())
        else:
            rt_new = {seg:None for seg in self.segs}
            r_new = RotationTable(table=rt_new, meta=self.meta.copy(), t=[])
        return r_new

    def read_from_STO(self, path):
        self.meta = {}
        self.rtable = {}
        with open(path) as f:
            for l in f:
                line = l.strip()
                if line == 'endheader':
                    break
                else:
                    meta = line.split('=')
                    self.meta[meta[0]] = meta[1]
            # read the header line
            segs = f.readline().strip().split('\t')
            segs.pop(0) # remove 'time'
            # read the rest
            p = pd.read_csv(f, header=None, sep=r'\t|,',
                            skipinitialspace=True, dtype=np.float64, engine='python')
        arr = p.values
        self.t = np.squeeze(arr[:,0:1]) # store time separately
        arr = arr[:,1:] # data without time
        # convert from quaternions to rotation
        for nSeg, seg in enumerate(segs):
            quats = arr[:,nSeg*4:(nSeg+1)*4]
            # shuffle the quaternion vector to be compatible with scipy notation
            quats = quats[:,[1,2,3,0]]
            # convert to scipy rotation object
            rot = R.from_quat(quats)
            self[seg] = rot

    def write_to_STO(self, path):
        data_txt = []
        t_csv = np.expand_dims([f'{tt:f}' for tt in self.t], 1)
        data_txt.append(t_csv)
        for seg in self:
            q = self[seg].as_quat()
            # shuffle the quaternion vector to be compatible with OpenSim notation
            q = q[:,[3,0,1,2]]
            seg_csv = np.expand_dims([','.join([f'{el:.10f}' for el in row]) for row in q], 1)
            data_txt.append(seg_csv)
        data_txt = np.hstack(data_txt)
        hdr = '\n'.join([f'{k}={self.meta[k]}' for k in self.meta]) + '\n' + 'endheader\n'
        hdr += 'time\t' + '\t'.join([f'{seg}' for seg in self])
        np.savetxt(path, data_txt, fmt='%s', delimiter='\t', header=hdr, comments='')

    def as_array(self, repr=R.as_quat, *args, **kwargs):
        arr = np.hstack([repr(self[seg], *args, **kwargs) for seg in self.segs])
        return arr

    def from_array(self, arr, repr=R.from_quat, *args, **kwargs):
        rt_new = self.copy()
        for nSeg, seg in enumerate(self.segs):
            if repr == R.from_quat:
                r = repr(arr[:,nSeg*4:(nSeg+1)*4])
            else:
                r = repr(arr[:,nSeg*3:(nSeg+1)*3], *args, **kwargs)
            rt_new[seg] = r
        return rt_new

    def rotate(self, rot, seg=None, body=0):
        rt_new = self.copy()
        rot = rot.as_matrix()
        if seg is not None:
            segs = [seg]
        else:
            segs = self.segs
        for s in segs:
            r = (self[s].as_matrix() @ rot) if body \
                                    else (rot @ self[s].as_matrix())
            rt_new[s] = R.from_matrix(r)
        return rt_new

    def parse(self, markers, rs_len):
        arr = self.as_array(R.as_matrix)
        arr_parsed = parse_array(arr, markers, rs_len)
        t_parsed = parse_array(self.t, markers, rs_len)
        rt_parsed = self.from_array(arr_parsed, R.from_matrix)
        rt_parsed.t = t_parsed
        return rt_parsed

    def vel(self, body=0):
        '''
        Compute the velocity of the rotation table (in the body or global frame).
        Velocity is computed as the difference between two consecutive rotations.
        such that R[i+1] = R[i] * vel[i] (body frame) or R[i+1] = vel[i] * R[i] (global frame).
        If R has N elements, vel will have N-1 elemnts.
        '''
        vel_rt = self.copy()
        for seg in vel_rt:
            r = vel_rt[seg][:-1] # discard the last row
            r_next = vel_rt[seg][1:]
            if body:
                vel_rt[seg] = r.inv() * r_next
            else:
                vel_rt[seg] = r_next * r.inv()
        vel_rt.t = vel_rt.t[:-1] # discard the last row
        return vel_rt

    def apply(self, fun, *args, **kwargs):
        arr = self.as_array()
        arr_new = fun(arr, *args, **kwargs)
        rt_new = self.from_array(arr_new)
        # !TODO! check length o.w. t might be invalid
        return rt_new

    def interpolate(self, t_new):
        '''
        Interpolate the rotation table to a new time vector.
        The interpolation is done using Akima method.
        '''
        r = self.as_array(R.as_matrix) # matrix array
        akima = Akima1DInterpolator(self.t, r, axis=0)
        r_new = akima(t_new, extrapolate=True)
        rt_new = self.from_array(r_new, R.from_matrix)
        rt_new.t = t_new.copy()
        return rt_new

    @property
    def segs(self):
        return self.rtable.keys()

    @property
    def nSeg(self):
        return len(self.segs)

    @property
    def nSample(self):
        return len(self)

    def __len__(self):
        return len(self[list(self.segs)[0]]) if self.segs else 0

    def __getitem__(self, item):
        if isinstance(item, slice):
            r = self.as_array()
            r = r[item.start:item.stop:item.step, :]
            rt_new = self.from_array(r)
            rt_new.t = self.t[item.start:item.stop:item.step]
            return rt_new
        else:
            return self.rtable[item]

    def __setitem__(self, item, val):
        self.rtable[item] = val

    def __iter__(self):
        return iter(self.rtable)


# import timeit
# cam_data = RotationTable(path_cam)

# timeit.timeit('cam_data.parse(r_strikes_cam, 200)', globals=globals(), number=200)