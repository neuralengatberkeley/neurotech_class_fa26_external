import numpy as np
from scipy import signal
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def downsample_kin_spike(hand_kinematics, fs, spike_times, bin_width=0.2, align_bins=False):
    """
    Downsample kinematics and bin spikes at a common time resolution.
    With align_bins=True, average matching windows and timestamp at bin ends.
    """
    downsample_ratio = int(round(bin_width*fs))
    if downsample_ratio<1 or not np.isclose(downsample_ratio,bin_width*fs):
        raise ValueError("Bin width must be a positive integer multiple of the kinematic sampling interval.")
    n_bins = len(hand_kinematics)//downsample_ratio
    edges = np.arange(n_bins+1)*bin_width
    Y_spikes = np.column_stack([np.diff(np.searchsorted(times,edges,side="left"))/bin_width for times in spike_times])

    if align_bins:
        X_kin = hand_kinematics[:n_bins*downsample_ratio].reshape(
            n_bins,downsample_ratio,hand_kinematics.shape[1]).mean(axis=1)
        time_continuous = edges[1:]
    else:
        X = signal.resample_poly(hand_kinematics,1,downsample_ratio,axis=0) if downsample_ratio>1 else hand_kinematics.copy()
        X_kin = X[:n_bins]
        time_continuous = np.arange(n_bins)*bin_width

    return X_kin, Y_spikes, time_continuous

def plot_velocity_prediction(time_s,truth,prediction,title):
    scores = r2_score(truth,prediction,multioutput="raw_values")
    fig,axs = plt.subplots(2,1,figsize=(10,5),dpi=120,sharex=True)
    for column,ax in enumerate(axs):
        ax.plot(time_s,truth[:,column],label="True")
        ax.plot(time_s,prediction[:,column],label="Predicted",alpha=0.8)
        ax.text(0.02,0.9,f"R² = {scores[column]:.3f}",transform=ax.transAxes,
                bbox={"facecolor":"white","alpha":0.8,"edgecolor":"none"})
        ax.set(xlim=(time_s[0],time_s[0]+60),ylabel=f"{'XY'[column]} velocity (cm/s)")
    axs[0].set_title(title)
    axs[0].legend(loc="upper right")
    axs[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return scores

# Kalman-filter training; X and Y use (variables, time) orientation.
def train_kalman_filter(Y,X):
    A = np.linalg.lstsq(X[:,:-1].T,X[:,1:].T,rcond=None)[0].T
    H = np.linalg.lstsq(X.T,Y.T,rcond=None)[0].T
    state_residual = X[:,1:]-A@X[:,:-1]
    observation_residual = Y-H@X
    W = state_residual@state_residual.T/(X.shape[1]-1)
    Q = observation_residual@observation_residual.T/X.shape[1]
    return A,W,H,Q

def run_kalman_forward(A,W,H,Q,Y,initial_state):
    n_states = A.shape[0]
    prediction = np.empty((n_states,Y.shape[1]))
    prediction[:,0] = initial_state
    covariance = np.zeros((n_states,n_states))
    identity = np.eye(n_states)
    gains = []
    for sample in range(1,Y.shape[1]):
        prior_state = A@prediction[:,sample-1]
        prior_covariance = A@covariance@A.T+W
        gain = prior_covariance@H.T@np.linalg.pinv(H@prior_covariance@H.T+Q)
        gains.append(gain)
        prediction[:,sample] = prior_state+gain@(Y[:,sample]-H@prior_state)
        covariance = (identity-gain@H)@prior_covariance
        covariance = (covariance+covariance.T)/2
    return prediction, gains
