import os
import json
import numpy as np
from collections import defaultdict

ROUND_DIGIT = 4

def make_stats(eval_dir):
    all_scores = []
    qid_scores = defaultdict(list)
    scenario_scores = defaultdict(list)
    scenario_qid_scores = defaultdict(lambda: defaultdict(list))
    
    for folder in sorted(os.listdir(eval_dir)):
        folder_path = os.path.join(eval_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        
        scenario_name = folder.split('_')[0]
        
        json_files = sorted(
            [f for f in os.listdir(folder_path) if f.endswith('.json')],
            key=lambda x: int(os.path.splitext(x)[0])
        )
        
        for json_file in json_files:
            json_path = os.path.join(folder_path, json_file)
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data = data['QA']
                for category in data:
                    for item in data[category]:
                        qid = item["qid"]
                        score = item["score"]
                        
                        all_scores.append(score)
                        qid_scores[qid].append(score)
                        scenario_scores[scenario_name].append(score)
                        scenario_qid_scores[scenario_name][qid].append(score)
    
    overall_avg = round(np.mean(all_scores), ROUND_DIGIT) if all_scores else 0
    
    qid_avg_scores = {qid: round(np.mean(scores), ROUND_DIGIT) for qid, scores in qid_scores.items()}
    
    scenario_avg_scores = {scenario: round(np.mean(scores), ROUND_DIGIT) for scenario, scores in scenario_scores.items()}
    
    scenario_qid_avg_scores = {
        scenario: {qid: round(np.mean(scores), ROUND_DIGIT) for qid, scores in qid_dict.items()}
        for scenario, qid_dict in scenario_qid_scores.items()
    }
    
    return {
        "overall_avg": overall_avg,
        "qid_avg_scores": qid_avg_scores,
        "scenario_avg_scores": scenario_avg_scores,
        "scenario_qid_avg_scores": scenario_qid_avg_scores
    }

def save_stats(eval_dir):
    stats = make_stats(eval_dir)
    stats_path = os.path.join(eval_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)
    print(f"Statistics saved to {stats_path}")

if __name__ == "__main__":
    save_stats("./eval_outputs")