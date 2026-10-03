import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import defaultdict
import os

# Function to parse a log file
def parse_log(file_path):
    matchpoint_dist = {}
    textformat_dist = {}
    combinations = defaultdict(int)

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find sections
    in_mp = False
    in_tf = False
    in_combo = False
    for line in lines:
        line = line.strip()
        if '--- MatchPoint 分數分佈 ---' in line:
            in_mp = True
            in_tf = False
            in_combo = False
            continue
        elif '--- TextFormat 分數分佈 ---' in line:
            in_mp = False
            in_tf = True
            in_combo = False
            continue
        elif '--- 分數組合分佈' in line:
            in_mp = False
            in_tf = False
            in_combo = True
            continue
        elif line.startswith('---') or line.startswith('=') or not line:
            in_mp = False
            in_tf = False
            in_combo = False
            continue

        if in_mp and '分數' in line and '次' in line:
            # Use regex to extract score and count
            match = re.search(r'分數 ([\d.]+): (\d+) 次', line)
            if match:
                score = float(match.group(1))
                count = int(match.group(2))
                matchpoint_dist[score] = count

        if in_tf and '分數' in line and '次' in line:
            match = re.search(r'分數 ([\d.]+): (\d+) 次', line)
            if match:
                score = float(match.group(1))
                count = int(match.group(2))
                textformat_dist[score] = count

        if in_combo and '組合' in line and '次' in line:
            # 組合 (9.5, 9.8): 173 次
            match = re.search(r'組合 \(([\d.]+), ([\d.]+)\): (\d+) 次', line)
            if match:
                mp = float(match.group(1))
                tf = float(match.group(2))
                count = int(match.group(3))
                combinations[(mp, tf)] = count

    return matchpoint_dist, textformat_dist, combinations

# Files to process
files = {
    'analysis_tem0.9.log': 'results/model_predictions_full3_adv2_tem0.9_p0.95.jsonl',
    'analysis_tem0.85.log': 'results/model_predictions_full3_adv2_tem0.85_p0.95.jsonl',
    'analysis_tem0.95.log': 'results/model_predictions_full3_adv2_tem0.95_p0.95.jsonl',
}

# Proceed

for log_file, model_file in files.items():
    if not os.path.exists(log_file):
        continue
    mp_dist, tf_dist, combos = parse_log(log_file)
    print(f"Processing {log_file}: MP keys {list(mp_dist.keys())}, TF keys {list(tf_dist.keys())}, Combos {len(combos)}")

    # Plot MatchPoint distribution
    if mp_dist:
        plt.figure(figsize=(10, 6))
        scores = list(mp_dist.keys())
        counts = list(mp_dist.values())
        plt.bar(scores, counts, color='skyblue')
        plt.xlabel('MatchPoint Score')
        plt.ylabel('Count')
        plt.title(f'MatchPoint Score Distribution for {model_file}')
        plt.xticks(scores)
        plt.savefig(f'{log_file.replace(".log", "")}_matchpoint.png')
        plt.close()
        print(f"Saved {log_file.replace('.log', '')}_matchpoint.png")
    else:
        print(f"No MP data for {log_file}")

    # Plot TextFormat distribution
    if tf_dist:
        plt.figure(figsize=(10, 6))
        scores = list(tf_dist.keys())
        counts = list(tf_dist.values())
        plt.bar(scores, counts, color='lightgreen')
        plt.xlabel('TextFormat Score')
        plt.ylabel('Count')
        plt.title(f'TextFormat Score Distribution for {model_file}')
        plt.xticks(scores)
        plt.savefig(f'{log_file.replace(".log", "")}_textformat.png')
        plt.close()
        print(f"Saved {log_file.replace('.log', '')}_textformat.png")
    else:
        print(f"No TF data for {log_file}")

    # Plot combinations as heatmap
    if combos:
        mp_scores = sorted(set(mp for mp, tf in combos.keys()))
        tf_scores = sorted(set(tf for mp, tf in combos.keys()))
        data = [[combos.get((mp, tf), 0) for tf in tf_scores] for mp in mp_scores]

        plt.figure(figsize=(10, 8))
        sns.heatmap(data, annot=True, fmt='d', xticklabels=tf_scores, yticklabels=mp_scores, cmap='Blues')
        plt.xlabel('TextFormat Score')
        plt.ylabel('MatchPoint Score')
        plt.title(f'Score Combinations Heatmap for {model_file}')
        plt.savefig(f'{log_file.replace(".log", "")}_combinations.png')
        plt.close()
        print(f"Saved {log_file.replace('.log', '')}_combinations.png")
    else:
        print(f"No combo data for {log_file}")

print("Charts generated.")