import csv
import random
import os
import re
import requests
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# API设置
OLLAMA_BASE_URL = "http://localhost:11434/api"

# 固定文件名
ONLY_POSTS = "only_posts.txt"
RIGHT_LABELS = "right_labels.txt"
SENTIMENT_OUTPUT = "sentiment_labels.txt"
BASE_ACCURACY_OUTPUT = "base_accuracy_metrics.txt"
ENHANCED_ACCURACY_OUTPUT = "enhanced_accuracy_metrics.txt"
COMPARISON_OUTPUT = "accuracy_comparison.txt"
COMPARISON_PLOT = "accuracy_comparison.png"


def prepare_data(input_file: str, target_dir: str):
    """准备分析数据，生成帖子文件和标签文件"""
    print("\n" + "=" * 50)
    print("数据准备阶段")
    print("=" * 50)

    # 创建目标文件夹
    os.makedirs(target_dir, exist_ok=True)

    dev_posts_path = os.path.join(target_dir, ONLY_POSTS)
    dev_labels_path = os.path.join(target_dir, RIGHT_LABELS)

    # 读取并处理数据
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter='\t')
        data = {"fake": [], "real": []}
        for row in reader:
            label = row["label"].strip().lower()
            if label in data:
                data[label].append(row["post_text"])

    # 检查数据量并采样
    min_samples = min(len(data["fake"]), len(data["real"]))
    if min_samples < 20:
        raise ValueError(f"某个标签的数据量不足20条 (fake: {len(data['fake'])}, real: {len(data['real'])})")

    print(f"从每个标签各抽取20条样本 (共40条数据)")

    # 随机抽样并合并
    sampled_data = [{
        "text": text,
        "label": 0 if label == "fake" else 1
    } for label in ["fake", "real"] for text in random.sample(data[label], 20)]

    random.shuffle(sampled_data)  # 打乱顺序

    # 保存结果
    with open(dev_posts_path, "w", encoding="utf-8") as p, \
            open(dev_labels_path, "w", encoding="utf-8") as l:
        for item in sampled_data:
            p.write(item["text"] + "\n")
            l.write(str(item["label"]) + "\n")

    print(f"文本保存至: {dev_posts_path}")
    print(f"标签保存至: {dev_labels_path}")
    return dev_posts_path, dev_labels_path


def generate_text(prompt: str, model: str = "deepseek-r1:1.5b",
                  temperature: float = 0.1, max_tokens: int = 2) -> str:
    """调用Ollama API生成文本"""
    url = f"{OLLAMA_BASE_URL}/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.RequestException as e:
        print(f"API请求错误: {e}")
        return ""
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return ""


def verify_news(text: str) -> str:
    """验证新闻真假"""
    prompt = f"""请判断以下社交媒体内容的真实性，严格依据以下标准仅输出0（假）或1（真）：
    内容：{text}
    判断标准：
    1. 事实要素（时间/地点/人物等是否准确）
    2. 逻辑一致性（是否存在夸张/矛盾/不可验证表述）
    3. 信源可靠性（是否来自权威媒体/认证账号）
    仅回复0或1，禁止附加任何内容"""
    prediction = generate_text(prompt)
    return '1' if '1' in prediction else '0'


def analyze_sentiment(input_file: str, output_dir: str):
    """进行情感分析"""
    print("\n" + "=" * 50)
    print("情感分析阶段")
    print("=" * 50)

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件 '{input_file}' 不存在")

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, SENTIMENT_OUTPUT)

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]

        print(f"读取了 {len(texts)} 条文本")

        with open(output_file, 'w', encoding='utf-8') as f_out:
            for text in tqdm(texts, desc="分析情感"):
                prompt = f"""请精准判断以下内容的情感倾向，严格按标准输出对应数字：
                内容：{text}
                - 积极（含褒义词/支持态度）→ 1
                - 中性（无明显倾向）→ 0
                - 消极（含贬义词/反对态度）→ -1
                仅输出-1/0/1中的单一数字，禁止附加任何文字"""

                prediction = generate_text(prompt, max_tokens=3)

                # 只提取数字，忽略其他文本
                if '1' in prediction:
                    sentiment = '1'
                elif '-1' in prediction:
                    sentiment = '-1'
                else:
                    sentiment = '0'

                f_out.write(f"{sentiment}\n")

        print(f"\n情感分析结果已保存到: {output_file}")
        return output_file

    except Exception as e:
        print(f"情感分析错误: {e}")
        raise


def verify_news_with_sentiment(text: str, sentiment: int) -> str:
    """结合情感倾向验证新闻真假"""
    sentiment_map = {1: "积极", 0: "中性", -1: "消极"}
    prompt = f"""请结合情感倾向判断以下内容的真实性（情感倾向: {sentiment_map.get(sentiment, '未知')}）：
    内容：{text}
    判断规则：
    1. 消极内容：重点核查事实准确性（时间/地点/人物等要素）
    2. 积极内容：重点检测表述是否存在夸张/不可验证
    3. 中性内容：重点评估事实完整性与逻辑一致性
    请严格仅输出0（假）或1（真），无需附加任何文字"""

    prediction = generate_text(prompt)
    return '1' if '1' in prediction else '0'


def calculate_accuracy(predictions, ground_truth):
    """计算准确率指标"""
    total = len(predictions)
    correct = sum(p == gt for p, gt in zip(predictions, ground_truth))
    true_total = sum(1 for gt in ground_truth if gt == '1')
    true_correct = sum(1 for p, gt in zip(predictions, ground_truth) if p == gt == '1')
    fake_total = total - true_total
    fake_correct = sum(1 for p, gt in zip(predictions, ground_truth) if p == gt == '0')

    accuracy = correct / total if total > 0 else 0
    accuracy_true = true_correct / true_total if true_total > 0 else 0
    accuracy_fake = fake_correct / fake_total if fake_total > 0 else 0

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "true_total": true_total,
        "true_correct": true_correct,
        "accuracy_true": accuracy_true,
        "fake_total": fake_total,
        "fake_correct": fake_correct,
        "accuracy_fake": accuracy_fake
    }


def run_accuracy_analysis(input_file: str, label_file: str, output_dir: str,
                          use_sentiment: bool = False, sentiment_file: str = None):
    """运行准确率分析"""
    mode = "结合情感分析" if use_sentiment else "基础版"
    print(f"\n" + "=" * 50)
    print(f"新闻真假判别准确率分析 ({mode})")
    print("=" * 50)

    # 验证文件路径
    for file_path in [input_file, label_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 '{file_path}' 不存在")

    if use_sentiment and sentiment_file and not os.path.exists(sentiment_file):
        raise FileNotFoundError(f"情感文件 '{sentiment_file}' 不存在")

    os.makedirs(output_dir, exist_ok=True)

    # 确定输出文件名
    if use_sentiment:
        output_file = os.path.join(output_dir, ENHANCED_ACCURACY_OUTPUT)
    else:
        output_file = os.path.join(output_dir, BASE_ACCURACY_OUTPUT)

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]

        with open(label_file, 'r', encoding='utf-8') as f:
            ground_truth = [line.strip() for line in f if line.strip()]

        if len(texts) != len(ground_truth):
            raise ValueError(f"文本数量({len(texts)})与标签数量({len(ground_truth)})不一致")

        sentiments = []
        if use_sentiment and sentiment_file:
            with open(sentiment_file, 'r', encoding='utf-8') as f:
                sentiments = [line.strip() for line in f if line.strip()]

            if len(texts) != len(sentiments):
                raise ValueError(f"文本数量({len(texts)})与情感标签数量({len(sentiments)})不一致")

        print(f"\n开始处理 {len(texts)} 条新闻数据...")

        predictions = []
        if use_sentiment:
            for i, (text, sentiment) in enumerate(tqdm(zip(texts, sentiments), desc="预测", total=len(texts))):
                try:
                    sentiment_int = int(sentiment)
                    predictions.append(verify_news_with_sentiment(text, sentiment_int))
                except ValueError:
                    predictions.append(verify_news_with_sentiment(text, 0))
        else:
            for text in tqdm(texts, desc="预测"):
                predictions.append(verify_news(text))

        metrics = calculate_accuracy(predictions, ground_truth)

        with open(output_file, 'w', encoding='utf-8') as f:
            title = "结合情感分析的" if use_sentiment else "基础版"
            f.write(f"{title}新闻真假判别准确率分析报告\n{'=' * 50}\n")
            f.write(f"总新闻数量: {metrics['total']}\n")
            f.write(f"预测正确的新闻数量: {metrics['correct']}\n")
            f.write(f"整体准确率: {metrics['accuracy']:.4f}\n\n")
            f.write(f"真新闻数量: {metrics['true_total']}\n")
            f.write(f"预测正确的真新闻数量: {metrics['true_correct']}\n")
            f.write(f"真新闻准确率: {metrics['accuracy_true']:.4f}\n\n")
            f.write(f"假新闻数量: {metrics['fake_total']}\n")
            f.write(f"预测正确的假新闻数量: {metrics['fake_correct']}\n")
            f.write(f"假新闻准确率: {metrics['accuracy_fake']:.4f}\n")

        print(f"\n准确率分析结果已保存到: {output_file}")
        return output_file

    except Exception as e:
        print(f"准确率分析错误: {e}")
        raise


def compare_metrics(base_metrics_file: str, enhanced_metrics_file: str, output_dir: str):
    """比较准确率指标"""
    print("\n" + "=" * 50)
    print("结果比较阶段")
    print("=" * 50)

    for file_path in [base_metrics_file, enhanced_metrics_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 '{file_path}' 不存在")

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, COMPARISON_OUTPUT)
    output_plot = os.path.join(output_dir, COMPARISON_PLOT)

    def parse_metrics(file_path):
        metrics = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

                metrics['accuracy'] = float(re.search(r'整体准确率: (\d+\.\d+)', content).group(1))
                metrics['accuracy_true'] = float(re.search(r'真新闻准确率: (\d+\.\d+)', content).group(1))
                metrics['accuracy_fake'] = float(re.search(r'假新闻准确率: (\d+\.\d+)', content).group(1))

                match = re.search(r'(.+)新闻真假判别准确率分析报告', content)
                metrics['method'] = match.group(1).strip() if match else "未知方法"

        except Exception as e:
            print(f"解析文件 {file_path} 时出错: {e}")
            return None
        return metrics

    base = parse_metrics(base_metrics_file)
    enhanced = parse_metrics(enhanced_metrics_file)

    if not base or not enhanced:
        print("无法解析指标文件")
        return None

    improvements = {
        'accuracy': enhanced['accuracy'] - base['accuracy'],
        'accuracy_true': enhanced['accuracy_true'] - base['accuracy_true'],
        'accuracy_fake': enhanced['accuracy_fake'] - base['accuracy_fake']
    }

    # 保存比较结果
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("情感分析对新闻真假判别准确率的影响分析\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"{base['method']} 准确率:\n")
        f.write(f"  整体准确率: {base['accuracy']:.4f}\n")
        f.write(f"  真新闻准确率: {base['accuracy_true']:.4f}\n")
        f.write(f"  假新闻准确率: {base['accuracy_fake']:.4f}\n\n")

        f.write(f"{enhanced['method']} 准确率:\n")
        f.write(f"  整体准确率: {enhanced['accuracy']:.4f}\n")
        f.write(f"  真新闻准确率: {enhanced['accuracy_true']:.4f}\n")
        f.write(f"  假新闻准确率: {enhanced['accuracy_fake']:.4f}\n\n")

        f.write("准确率提升:\n")
        f.write(f"  整体准确率提升: {improvements['accuracy']:+.4f}\n")
        f.write(f"  真新闻准确率提升: {improvements['accuracy_true']:+.4f}\n")
        f.write(f"  假新闻准确率提升: {improvements['accuracy_fake']:+.4f}\n\n")

        # 总结主要发现
        overall_improved = improvements['accuracy'] > 0
        true_improved = improvements['accuracy_true'] > 0
        fake_improved = improvements['accuracy_fake'] > 0

        f.write("主要发现:\n")
        f.write(f"  - 整体准确率{'提高' if overall_improved else '降低'}了 {abs(improvements['accuracy']):.2%}\n")
        f.write(
            f"  - 真新闻识别准确率{'提高' if true_improved else '降低'}了 {abs(improvements['accuracy_true']):.2%}\n")
        f.write(
            f"  - 假新闻识别准确率{'提高' if fake_improved else '降低'}了 {abs(improvements['accuracy_fake']):.2%}\n\n")

        if overall_improved:
            f.write("结论: 结合情感分析显著提升了模型的真假新闻判别能力。\n")
        else:
            f.write("结论: 结合情感分析未显著提升模型的真假新闻判别能力，需进一步优化。\n")

    print(f"比较结果已保存到: {output_file}")

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 创建可视化图表
    labels = ['整体准确率', '真新闻准确率', '假新闻准确率']
    base_vals = [base['accuracy'], base['accuracy_true'], base['accuracy_fake']]
    enhanced_vals = [enhanced['accuracy'], enhanced['accuracy_true'], enhanced['accuracy_fake']]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.35
    rects1 = ax.bar(x - width / 2, base_vals, width, label=base['method'])
    rects2 = ax.bar(x + width / 2, enhanced_vals, width, label=enhanced['method'])

    # 添加数值标签
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    add_labels(rects1)
    add_labels(rects2)

    # 设置图表
    ax.set_ylabel('准确率')
    ax.set_title('情感分析对新闻真假判别准确率的影响')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # 添加准确率提升百分比
    for i, (b, e) in enumerate(zip(base_vals, enhanced_vals)):
        if b != 0:
            improvement = (e - b) / b * 100
            ax.text(x[i], max(b, e) + 0.01, f"{improvement:+.2f}%",
                    ha='center', va='bottom', color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_plot)
    print(f"可视化图表已保存到: {output_plot}")

    return output_file


def run_full_process():
    """运行完整的处理流程"""
    print("本流程将按顺序执行以下步骤:")
    print("1. 数据准备（生成分析样本）")
    print("2. 新闻真假判别")
    print("3. 情感分析")
    print("4. 新闻真假判别（结合情感分析）")
    print("5. 结果比较")
    print("=" * 50)

    # 获取输入参数
    input_file = input("请输入原始新闻数据文件路径: ").strip()
    output_dir = input("请输入结果输出目录: ").strip()

    try:
        # 步骤1: 数据准备
        print("\n>> 步骤1: 数据准备")
        posts_file, labels_file = prepare_data(input_file, output_dir)

        # 步骤2: 基础版分析
        print("\n>> 步骤2: 基础版真假判别")
        base_metrics_file = run_accuracy_analysis(
            input_file=posts_file,
            label_file=labels_file,
            output_dir=output_dir,
            use_sentiment=False
        )

        # 步骤3: 情感分析
        print("\n>> 步骤3: 情感分析")
        sentiment_file = analyze_sentiment(
            input_file=posts_file,
            output_dir=output_dir
        )

        # 步骤4: 情感增强版分析
        print("\n>> 步骤4: 结合情感分析的真假判别")
        enhanced_metrics_file = run_accuracy_analysis(
            input_file=posts_file,
            label_file=labels_file,
            output_dir=output_dir,
            use_sentiment=True,
            sentiment_file=sentiment_file
        )

        # 步骤5: 结果比较
        print("\n>> 步骤5: 结果比较")
        compare_metrics(
            base_metrics_file=base_metrics_file,
            enhanced_metrics_file=enhanced_metrics_file,
            output_dir=output_dir
        )

        print("\n" + "=" * 50)
        print("所有处理步骤已完成！")
        print(f"结果文件保存在: {os.path.abspath(output_dir)}")
        print("=" * 50)

    except Exception as e:
        print(f"\n处理过程中发生错误: {e}")
    finally:
        input("按Enter键退出程序...")


if __name__ == "__main__":
    run_full_process()