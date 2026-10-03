import json
import re
import logging
from collections import Counter

def parse_scores_from_string(text):
    """
    從模型輸出的字串中，穩健地提取 MatchPoint 和 TextFormat 的分數。
    """
    try:
        match_point_match = re.search(r"MatchPoint:\s*([0-9\.]+)", text)
        match_point = float(match_point_match.group(1)) if match_point_match else None

        text_format_match = re.search(r"TextFormat:\s*([0-9\.]+)", text)
        text_format = float(text_format_match.group(1)) if text_format_match else None

        return match_point, text_format
    except (AttributeError, ValueError):
        return None, None

def check_format(text):
    """
    檢查輸出是否符合要求的格式結構。
    """
    # 檢查開頭
    starts_correctly = text.startswith("【給分原因】\n題文匹配度(MatchPoint):")
    # 檢查是否包含關鍵部分
    has_text_format_section = "文本格式(TextFormat):" in text
    has_score_section_header = "【分數】" in text
    
    return starts_correctly and has_text_format_section and has_score_section_header

def analyze_predictions(prediction_file):
    """
    讀取預測檔案，進行格式檢查與分數分佈統計。
    """
    # 配置 logging
    logging.basicConfig(
        filename='analysis_tem0.95.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        encoding='utf-8'
    )
    
    predictions = []
    with open(prediction_file, 'r', encoding='utf-8') as f:
        for line in f:
            predictions.append(json.loads(line))

    total_samples = len(predictions)
    format_correct_count = 0
    
    match_points = []
    text_formats = []
    score_pairs = []
    
    unparsable_count = 0

    for item in predictions:
        # 從 'model_resp' 欄位讀取模型生成的完整輸出
        model_output = item.get('model_resp', '')

        # 1. 格式檢查
        if check_format(model_output):
            format_correct_count += 1
        
        # 2. 分數解析
        mp, tf = parse_scores_from_string(model_output)
        
        if mp is not None and tf is not None:
            match_points.append(mp)
            text_formats.append(tf)
            score_pairs.append((mp, tf))
        else:
            unparsable_count += 1

    # --- 產生最終報告 ---
    logging.info("\n==================== 模型輸出分析報告 ====================")
    logging.info(f"檔案: {prediction_file}")
    logging.info(f"總樣本數: {total_samples}")
    
    logging.info("\n--- 格式符合度分析 ---")
    format_percentage = (format_correct_count / total_samples) * 100 if total_samples > 0 else 0
    logging.info(f"符合指定格式的樣本數: {format_correct_count} / {total_samples} ({format_percentage:.2f}%)")
    
    logging.info("\n--- 分數解析成功率 ---")
    parsable_percentage = ((total_samples - unparsable_count) / total_samples) * 100 if total_samples > 0 else 0
    logging.info(f"可成功解析分數的樣本數: {total_samples - unparsable_count} / {total_samples} ({parsable_percentage:.2f}%)")

    if not match_points:
        logging.info("\n未解析到任何有效分數，無法進行分佈統計。")
        logging.info("========================================================\n")
        return

    logging.info("\n--- MatchPoint 分數分佈 ---")
    mp_counts = Counter(match_points)
    for score, count in sorted(mp_counts.items()):
        logging.info(f"分數 {score}: {count} 次")

    logging.info("\n--- TextFormat 分數分佈 ---")
    tf_counts = Counter(text_formats)
    for score, count in sorted(tf_counts.items()):
        logging.info(f"分數 {score}: {count} 次")

    logging.info("\n--- 分數組合分佈 (MatchPoint, TextFormat) ---")
    pair_counts = Counter(score_pairs)
    # 按照出現次數從高到低排序，方便查看最常見的組合
    for pair, count in sorted(pair_counts.items(), key=lambda item: item[1], reverse=True):
        logging.info(f"組合 {pair}: {count} 次")

    logging.info("========================================================\n")


if __name__ == '__main__':
    # === 使用方法 ===
    # 請將下面的檔名替換成實際的預測檔路徑
    prediction_file_to_analyze = 'results/model_predictions_full3_adv2_tem0.95_p0.95.jsonl'
    
    analyze_predictions(prediction_file_to_analyze)