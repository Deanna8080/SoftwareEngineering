import sys
import re
import math
from collections import Counter


def read_file(path):
    """按 UTF-8 读取文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def normalize(text):
    """
    文本归一化：
    只保留汉字、英文字母、数字。
    英文统一转小写。
    """
    if text is None:
        return ""
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text).lower()


def ngram_counter(text, n=2):
    """生成字符 n-gram 频率。"""
    if not text:
        return Counter()

    if n <= 0:
        raise ValueError("n 必须大于 0")

    if len(text) < n:
        return Counter({text: 1})

    return Counter(text[i:i + n] for i in range(len(text) - n + 1))


def cosine_similarity(text_a, text_b, n=2):
    """使用余弦相似度计算两个文本的相似度，范围 0.00 ~ 1.00。"""
    norm_a = normalize(text_a)
    norm_b = normalize(text_b)

    freq_a = ngram_counter(norm_a, n)
    freq_b = ngram_counter(norm_b, n)

    if not freq_a or not freq_b:
        return 0.0

    dot = 0.0
    for key, value_a in freq_a.items():
        value_b = freq_b.get(key, 0)
        dot += value_a * value_b

    norm_a_value = math.sqrt(sum(v * v for v in freq_a.values()))
    norm_b_value = math.sqrt(sum(v * v for v in freq_b.values()))

    if norm_a_value == 0 or norm_b_value == 0:
        return 0.0

    return dot / (norm_a_value * norm_b_value)


def main():
    if len(sys.argv) != 4:
        print("用法: python main.py <原文文件> <抄袭版文件> <答案文件>")
        sys.exit(2)

    original_path = sys.argv[1]
    copied_path = sys.argv[2]
    answer_path = sys.argv[3]

    try:
        original_text = read_file(original_path)
        copied_text = read_file(copied_path)

        score = cosine_similarity(original_text, copied_text, n=2)

        with open(answer_path, "w", encoding="utf-8") as f:
            f.write(f"{score:.2f}")

        print(f"重复率: {score:.2f}")

    except Exception as e:
        print(f"程序执行失败: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()