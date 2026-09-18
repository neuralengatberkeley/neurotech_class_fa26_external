import numpy as np

GOCUE_CODE = 5
LEAVE_CENTER_CODE = 6
AT_TARGET_CODE = 7
REWARD_CODE = 9
TARGET_OFFSET = 63

def trial_align_events(events,event_times,align_code,num_before=3,num_after=3):
    indices = np.flatnonzero(events==align_code)
    indices = indices[(indices>=num_before)&(indices+num_after<len(events))]
    windows = indices[:,None]+np.arange(-num_before,num_after+1)
    return events[windows],event_times[windows]

# A shared trial list keeps kinematics, neural data, and labels aligned.
def epoch_trials(events, event_times, hand_kinematics, fs_kinematics):
    trial_events,trial_event_times = trial_align_events(events,event_times,LEAVE_CENTER_CODE)
    valid_trials = (
        np.isin(trial_events[:,0],np.arange(64,72))
        &(trial_events[:,2]==GOCUE_CODE)
        &(trial_events[:,4]==AT_TARGET_CODE)
    )
    recording_stop = len(hand_kinematics)/fs_kinematics
    valid_trials &= (trial_event_times[:,2]>=1.5)&(trial_event_times[:,3]+1.5<recording_stop)
    trial_events = trial_events[valid_trials]
    trial_event_times = trial_event_times[valid_trials]
    reach_direction = trial_events[:,0]-TARGET_OFFSET
    align_time_lc = trial_event_times[:,3]
    align_time_go = trial_event_times[:,2]

    return trial_events, reach_direction, align_time_lc, align_time_go
