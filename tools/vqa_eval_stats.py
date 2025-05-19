import os
import json
import argparse
import yaml
from collections import defaultdict

def process_run_eval(checkpoint_path, eval_base):
    with open(checkpoint_path, 'r') as f:
        cp = json.load(f)
    records = cp['_checkpoint']['records']

    global_qid_scores = defaultdict(list)
    save_name_stats = {}

    for rec in records:
        name = rec['save_name']
        eval_dir = os.path.join(eval_base, name)
        save_qid_scores = defaultdict(list)
        if not os.path.exists(eval_dir):
            print(f"[ERROR] Eval folder not found: {eval_dir}")
            continue

        for fname in sorted(os.listdir(eval_dir)):
            if not fname.endswith('.json'):
                continue
            path = os.path.join(eval_dir, fname)
            with open(path, 'r') as f:
                data = json.load(f)

            qa_list = data.get('QA', {}).get('Category', [])
            for qa in qa_list:
                qid = qa.get('qid')
                score = qa.get('score')
                if qid is not None and score is not None:
                    save_qid_scores[qid].append(score)
                    global_qid_scores[qid].append(score)

        save_name_stats[name] = {
            str(qid): {
                'average_score': sum(scores) / len(scores),
                'count': len(scores)
            } for qid, scores in save_qid_scores.items()
        }

    global_stats = {
        str(qid): {
            'average_score': sum(scores) / len(scores),
            'count': len(scores)
        } for qid, scores in global_qid_scores.items()
    }

    out_path = os.path.join(os.path.dirname(checkpoint_path),
                            f'vqa_eval_{os.path.basename(checkpoint_path)}')
    with open(out_path, 'w') as wf:
        json.dump({
            'per_scenario': save_name_stats,
            'overall': global_stats
        }, wf, indent=2)

    print(f'Wrote VQA evaluation results to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate VQA scores based on checkpoint and eval data')
    parser.add_argument('-f', '--config', required=True, help='YAML config path listing runs')
    args = parser.parse_args()

    with open(args.config, 'r') as yf:
        cfg = yaml.safe_load(yf)
    for entry in cfg.get('runs', []):
        ckpt = entry['checkpoint']
        eval_base = entry['eval_dir']
        process_run_eval(ckpt, eval_base)
