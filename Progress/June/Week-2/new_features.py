import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def make_sequence_summary_features(df, demographics_df=None):
    """
    Create comprehensive features from sensor sequences
    """
    features = []
    
    # Group by sequence_id to create sequence-level features
    for seq_id, group in df.groupby('sequence_id'):
        seq_features = {'sequence_id': seq_id}
        columns = set(group.columns)
        
        # Basic sequence info
        seq_features['sequence_length'] = len(group)
        seq_features['subject'] = group['subject'].iloc[0]
        
        # Add demographics if available
        if (demographics_df is not None) and (not demographics_df.empty):
            subject_demo = demographics_df[ demographics_df['subject'] == seq_features['subject'] ]
            if not subject_demo.empty:
                seq_features['adult_child'] = subject_demo['adult_child'].iloc[0]
                seq_features['age'] = subject_demo['age'].iloc[0]
                seq_features['sex'] = subject_demo['sex'].iloc[0]
                seq_features['handedness'] = subject_demo['handedness'].iloc[0]
                seq_features['height_cm']  = subject_demo['height_cm'].iloc[0]
                seq_features['shoulder_to_wrist_cm'] = subject_demo['shoulder_to_wrist_cm'].iloc[0]
                seq_features['elbow_to_wrist_cm']    = subject_demo['elbow_to_wrist_cm'].iloc[0]
            else:
                # Set default values if demographics not found
                seq_features['adult_child'] = -1
                seq_features['age'] = -1
                seq_features['sex'] = -1
                seq_features['handedness'] = -1
                seq_features['height_cm'] = -1
                seq_features['shoulder_to_wrist_cm'] = -1
                seq_features['elbow_to_wrist_cm'] = -1
        else:
            # Set default values if demographics not available
            seq_features['adult_child'] = -1
            seq_features['age'] = -1
            seq_features['sex'] = -1
            seq_features['handedness'] = -1
            seq_features['height_cm'] = -1
            seq_features['shoulder_to_wrist_cm'] = -1
            seq_features['elbow_to_wrist_cm'] = -1
        
        # Behavior phase encoding (if available)
        if 'behavior' in columns:
            behavior_counts = group['behavior'].value_counts()
            for behavior in ['Transition', 'Pause', 'Gesture']:
                seq_features[f'{behavior.lower()}_count'] = behavior_counts.get(behavior, 0)
                seq_features[f'{behavior.lower()}_ratio'] = behavior_counts.get(behavior, 0) / len(group)
        else:
            # Set default values if behavior column is not available
            for behavior in ['Transition', 'Pause', 'Gesture']:
                seq_features[f'{behavior.lower()}_count'] = 0
                seq_features[f'{behavior.lower()}_ratio'] = 0
        
        # Statistical features for each sensor type
        sensor_groups = {
            'acc': ['acc_x', 'acc_y', 'acc_z'],
            'rot': ['rot_w', 'rot_x', 'rot_y', 'rot_z'],
            'thm': ["thm_1", "thm_2", "thm_3", "thm_4", "thm_5"],
            'tof': [f"tof_{i}_v{j}" for i in range(1,6) for j in range(0,64)]
        }
        
        for sensor_type, cols in sensor_groups.items():
            available_cols = [col for col in cols if col in columns]
            if available_cols:
                sensor_data = group[available_cols].values        
                # Basic statistics
                seq_features[f'{sensor_type}_mean'] = np.mean(sensor_data)
                seq_features[f'{sensor_type}_std']  = np.std(sensor_data)
                seq_features[f'{sensor_type}_min']  = np.min(sensor_data)
                seq_features[f'{sensor_type}_max']  = np.max(sensor_data)
                seq_features[f'{sensor_type}_range']  = np.max(sensor_data) - np.min(sensor_data)
                seq_features[f'{sensor_type}_median'] = np.median(sensor_data)
                
                # Percentiles
                seq_features[f'{sensor_type}_q25'] = np.percentile(sensor_data, 25)
                seq_features[f'{sensor_type}_q75'] = np.percentile(sensor_data, 75)
                seq_features[f'{sensor_type}_iqr'] = np.percentile(sensor_data, 75) - np.percentile(sensor_data, 25)                
                
                # Signal characteristics
                seq_features[f'{sensor_type}_energy'] = np.sum(sensor_data**2)
                seq_features[f'{sensor_type}_rms'] = np.sqrt(np.mean(sensor_data**2))

                if sensor_type != "tof":
                    for col in available_cols:
                        sensor_data = group[col].values
                        seq_features[f'{col}_mean'] = np.mean(sensor_data)
                        seq_features[f'{col}_std']  = np.std(sensor_data)
                        seq_features[f'{col}_min']  = np.min(sensor_data)
                        seq_features[f'{col}_max']  = np.max(sensor_data)
                        seq_features[f'{col}_range']  = np.max(sensor_data) - np.min(sensor_data)
                        seq_features[f'{col}_median'] = np.median(sensor_data)
                    
                        # Percentiles
                        seq_features[f'{col}_q25'] = np.percentile(sensor_data, 25)
                        seq_features[f'{col}_q75'] = np.percentile(sensor_data, 75)
                        seq_features[f'{col}_iqr'] = np.percentile(sensor_data, 75) - np.percentile(sensor_data, 25)                
                
        # Specific features for IMU data (acceleration and rotation)
        if all(col in columns for col in ['acc_x', 'acc_y', 'acc_z']):
            acc_data = group[['acc_x', 'acc_y', 'acc_z']].values
            # Acceleration features
            acc_magnitude = np.sqrt(np.sum(acc_data**2, axis=1))
            jerk = np.nan_to_num(np.diff(acc_magnitude), nan=-666)
            seq_features['jerk_mean'] = np.mean(jerk)
            seq_features['jerk_std'] = np.std(jerk)
            seq_features['acc_magnitude_mean'] = np.mean(acc_magnitude)
            seq_features['acc_magnitude_std'] = np.std(acc_magnitude)
            seq_features['acc_magnitude_max'] = np.max(acc_magnitude)
            seq_features['acc_height_norm'] = seq_features['acc_magnitude_mean'] / max(seq_features['height_cm'], 1)
            seq_features['acc_shoulder_norm'] = seq_features['acc_magnitude_mean'] / max(seq_features['shoulder_to_wrist_cm'], 1)
            seq_features['acc_elbow_norm'] = seq_features['acc_magnitude_mean'] / max(seq_features['elbow_to_wrist_cm'], 1)
            seq_features['acc_xy_corr'] = spearmanr(group['acc_x'], group['acc_y'], nan_policy='omit').statistic
            seq_features['acc_yz_corr'] = spearmanr(group['acc_y'], group['acc_z'], nan_policy='omit').statistic
            seq_features['acc_xz_corr'] = spearmanr(group['acc_x'], group['acc_z'], nan_policy='omit').statistic
            seq_features["acc_x_cumsum"] = np.sum(group["acc_x"])
            seq_features["acc_y_cumsum"] = np.sum(group["acc_y"])
            seq_features["acc_z_cumsum"] = np.sum(group["acc_z"])
            
        # Rotational features
        rot_angle = 2*np.arccos(np.clip(group["rot_w"].values, -1.0, 1.0))
        angular_velocity = np.nan_to_num(np.diff(rot_angle), nan=-666)
        angular_acceleration = np.nan_to_num(np.diff(angular_velocity), nan=-666)
        seq_features['rot_wx_corr'] = np.nan_to_num(spearmanr(group['rot_w'], group['rot_x'], nan_policy='omit').statistic, nan=-666)
        seq_features['rot_wy_corr'] = np.nan_to_num(spearmanr(group['rot_w'], group['rot_y'], nan_policy='omit').statistic, nan=-666)
        seq_features['rot_wz_corr'] = np.nan_to_num(spearmanr(group['rot_w'], group['rot_z'], nan_policy='omit').statistic, nan=-666)
        seq_features['rot_xy_corr'] = np.nan_to_num(spearmanr(group['rot_x'], group['rot_y'], nan_policy='omit').statistic, nan=-666)
        seq_features['rot_xz_corr'] = np.nan_to_num(spearmanr(group['rot_x'], group['rot_z'], nan_policy='omit').statistic, nan=-666)
        seq_features['rot_yz_corr'] = np.nan_to_num(spearmanr(group['rot_y'], group['rot_z'], nan_policy='omit').statistic, nan=-666)
        seq_features['angular_velocity_mean'] = np.mean(angular_velocity)
        seq_features['angular_velocity_std'] = np.std(angular_velocity)
        seq_features['angular_accel_mean'] = np.mean(angular_acceleration)
        seq_features['angular_accel_std'] = np.std(angular_acceleration)
        seq_features["rot_angle_cumsum"] = np.sum(rot_angle)
        seq_features["rot_angle_mean"] = np.mean(rot_angle)
        seq_features["rot_angle_median"] = np.median(rot_angle)
        seq_features["rot_angle_std"]  = np.std(rot_angle)
        seq_features["rot_angle_min"]  = np.min(rot_angle)    
        seq_features["rot_angle_max"]  = np.max(rot_angle)
        seq_features["rot_angle_range"]  = np.max(rot_angle) - np.min(rot_angle)
        seq_features["rot_angle_q25"] = np.percentile(rot_angle, 25)
        seq_features["rot_angle_q75"] = np.percentile(rot_angle, 75)
        seq_features["rot_angle_iqr"] = np.percentile(rot_angle, 75) - np.percentile(rot_angle, 25)                
        seq_features['rot_angle_energy'] = np.sum(rot_angle**2)
        seq_features['rot_angle_rms'] = np.sqrt(np.mean(rot_angle**2))
        
        # Add target if available
        if 'encoded_gesture' in columns:
            seq_features['target'] = group['encoded_gesture'].iloc[0]
            seq_features['gesture'] = group['gesture'].iloc[0]
        
        features.append(seq_features)
    
    return pd.DataFrame(features)