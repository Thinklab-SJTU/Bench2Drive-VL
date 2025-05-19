import os
import json
import gzip
import argparse
import yaml
import numpy as np
import carla

def fix_rotation_order(rot):
    # input [pitch, roll, yaw] -> output [roll, pitch, yaw]
    return [rot[1], rot[0], rot[2]]

def get_rotation_matrix(rot):
    # rot is [roll, pitch, yaw] in degrees
    roll, pitch, yaw = np.radians(rot)
    
    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1]
    ])
    
    R_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0,             1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    R_roll = np.array([
        [1, 0,             0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    
    return R_yaw @ R_pitch @ R_roll

def transform_imu_to_world(acc_local, ang_local, rot):
    R = get_rotation_matrix(rot)
    acc_world = R @ np.array(acc_local)
    acc_world -= np.array([0, 0, 9.81])  # subtract gravity in world frame
    ang_world = R @ np.array(ang_local)
    return acc_world.tolist(), ang_world.tolist()

def process_run(checkpoint_path, anno_base, metrics_base):
    with open(checkpoint_path, 'r') as f:
        cp = json.load(f)
    records = cp['_checkpoint']['records']
    
    for rec in records:
        name = rec['save_name']
        anno_dir = os.path.join(anno_base, name, 'anno')
        out_dir = os.path.join(metrics_base, name)
        os.makedirs(out_dir, exist_ok=True)

        metric_info = {}
        for fname in sorted(os.listdir(anno_dir)):
            if not fname.endswith('.json.gz'):
                continue
            frame_id = os.path.splitext(os.path.splitext(fname)[0])[0]
            path = os.path.join(anno_dir, fname)
            with gzip.open(path, 'rt') as gf:
                data = json.load(gf)
            ego = None
            if isinstance(data, list):
                for item in data:
                    if item.get('class') == 'ego_vehicle':
                        ego = item
                        break
            else:
                ego = data.get('ego_vehicle') or data
            if ego is None:
                continue

            acc_local = data['acceleration']
            ang_local = data['angular_velocity']
            ego_entry = next(item for item in data['bounding_boxes'] if item['class'] == 'ego_vehicle')
            loc = ego_entry['location']  # [x, y, z]
            rot = fix_rotation_order(ego_entry['rotation'])  # [roll, pitch, yaw]

            acc_world, ang_world = transform_imu_to_world(acc_local, ang_local, rot)

            trans = carla.Transform(
                carla.Location(*loc),
                carla.Rotation(*rot)
            )
            fwd = trans.get_forward_vector()
            right = trans.get_right_vector()
            fwd = [fwd.x, fwd.y, fwd.z]
            right = [right.x, right.y, right.z]

            metric_info[int(frame_id)] = {
                'acceleration_IMU': acc_local,
                'angular_velocity_IMU': ang_local,
                'acceleration': acc_world,
                'angular_velocity': ang_world,
                'forward_vector': fwd,
                'right_vector': right,
                'location': loc,
                'rotation': rot
            }

        with open(os.path.join(out_dir, 'metric_info.json'), 'w') as wf:
            json.dump(metric_info, wf, indent=2)
        print(f'Wrote metrics for {name} -> {out_dir}/metric_info.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rebuild metric_info.json from anno files via YAML config')
    parser.add_argument('-f', '--config', required=True, help='YAML config path listing runs')
    args = parser.parse_args()

    with open(args.config, 'r') as yf:
        cfg = yaml.safe_load(yf)
    for entry in cfg.get('runs', []):
        ckpt = entry['checkpoint']
        anno_base = entry['anno_dir']
        metrics_base = entry['metrics_dir']
        process_run(ckpt, anno_base, metrics_base)
