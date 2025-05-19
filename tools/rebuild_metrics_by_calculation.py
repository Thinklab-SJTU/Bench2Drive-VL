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

def degrees_diff(a, b):
    # Compute smallest difference in degrees
    diff = a - b
    diff = (diff + 180) % 360 - 180
    return diff

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
        prev_loc = None
        prev_rot = None
        prev_vel = None

        for fname in sorted(os.listdir(anno_dir)):
            if not fname.endswith('.json.gz'):
                continue
            frame_id = int(os.path.splitext(os.path.splitext(fname)[0])[0])
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

            ego_entry = next(item for item in data['bounding_boxes'] if item['class'] == 'ego_vehicle')
            loc = ego_entry['location']  # [x, y, z]
            rot = fix_rotation_order(ego_entry['rotation'])  # [roll, pitch, yaw]

            trans = carla.Transform(
                carla.Location(*loc),
                carla.Rotation(*rot)
            )
            fwd = trans.get_forward_vector()
            right = trans.get_right_vector()
            fwd = [fwd.x, fwd.y, fwd.z]
            right = [right.x, right.y, right.z]

            if prev_loc is not None:
                dt = 0.1  # fixed timestep
                vel = [(loc[i] - prev_loc[i]) / dt for i in range(3)]
                ang_vel = [degrees_diff(rot[i], prev_rot[i]) / dt for i in range(3)]
                if prev_vel is not None:
                    acc = [(vel[i] - prev_vel[i]) / dt for i in range(3)]
                else:
                    acc = [0.0, 0.0, 0.0]
            else:
                vel = [0.0, 0.0, 0.0]
                acc = [0.0, 0.0, 0.0]
                ang_vel = [0.0, 0.0, 0.0]

            metric_info[frame_id] = {
                'velocity': vel,
                'acceleration': acc,
                'angular_velocity': ang_vel,
                'location': loc,
                'rotation': rot,
                'forward_vector': fwd,
                'right_vector': right
            }

            prev_loc = loc
            prev_rot = rot
            prev_vel = vel

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
