import os
import random
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import BertModel, BertTokenizer
import time
import csv
import requests
import json
from tqdm import tqdm
import logging

# 配置日志记录
logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ======================
# API设置
# ======================
OLLAMA_BASE_URL = "http://localhost:11434/api"

# ======================
# 数据预处理功能
# ======================
def prepare_data(input_file: str, target_dir: str, prefix: str):
    """准备分析数据，生成帖子文件和标签文件"""
    logging.info(f"\n{'=' * 50}\n准备数据: {prefix}\n{'=' * 50}")
    os.makedirs(target_dir, exist_ok=True)

    # 使用前缀创建文件名
    posts_path = os.path.join(target_dir, f"{prefix}_posts.txt")
    labels_path = os.path.join(target_dir, f"{prefix}_labels.txt")

    # 读取并处理数据
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter='\t')
        data = {"fake": [], "real": []}
        for row in reader:
            label = row["label"].strip().lower()
            if label in data:
                data[label].append(row["post_text"])

    # 检查数据量
    min_samples = min(len(data["fake"]), len(data["real"]))
    if min_samples < 20:
        logging.warning(f"某个标签的数据量不足20条 (fake: {len(data['fake'])}, real: {len(data['real'])}), 将使用所有可用样本")
        sample_size = min_samples
    else:
        sample_size = 20

    # 随机抽样并合并
    sampled_data = [{
        "text": text,
        "label": 0 if label == "fake" else 1
    } for label in ["fake", "real"] for text in random.sample(data[label], sample_size)]

    random.shuffle(sampled_data)  # 打乱顺序

    # 保存结果
    with open(posts_path, "w", encoding="utf-8") as p, \
            open(labels_path, "w", encoding="utf-8") as l:
        for item in sampled_data:
            p.write(item["text"] + "\n")
            l.write(str(item["label"]) + "\n")

    logging.info(f"文本保存至: {posts_path}")
    logging.info(f"标签保存至: {labels_path}")
    return posts_path, labels_path

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
        logging.error(f"API请求错误: {e}")
        return ""
    except json.JSONDecodeError as e:
        logging.error(f"JSON解析错误: {e}")
        return ""

def analyze_sentiment(input_file: str, output_dir: str, prefix: str):
    """进行情感分析"""
    logging.info(f"\n{'=' * 50}\n情感分析: {prefix}\n{'=' * 50}")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{prefix}_sentiment_labels.txt")

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]

        logging.info(f"读取了 {len(texts)} 条文本")

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

        logging.info(f"\n情感分析结果已保存到: {output_file}")
        return output_file

    except Exception as e:
        logging.error(f"情感分析错误: {e}")
        raise

def preprocess_data(train_file, val_file, output_dir):
    """数据预处理"""
    logging.info("\n" + "=" * 50)
    logging.info("开始数据预处理...")
    logging.info(f"- 训练集: {train_file}")
    logging.info(f"- 验证集: {val_file}")
    logging.info(f"- 输出目录: {output_dir}")
    logging.info("=" * 50)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 为训练集生成文件
    train_posts, train_labels = prepare_data(train_file, output_dir, "train")
    train_sentiment = analyze_sentiment(train_posts, output_dir, "train")

    # 为验证集生成文件
    val_posts, val_labels = prepare_data(val_file, output_dir, "val")
    val_sentiment = analyze_sentiment(val_posts, output_dir, "val")

    # 返回生成的文件路径
    output_files = {
        'train_text': train_posts,
        'train_labels': train_labels,
        'train_sentiment': train_sentiment,
        'val_text': val_posts,
        'val_labels': val_labels,
        'val_sentiment': val_sentiment
    }

    logging.info(f"\n数据预处理完成! 输出文件:")
    for name, path in output_files.items():
        logging.info(f"- {name}: {path}")

    return output_dir, output_files

# ======================
# 主题建模功能 (LDA)
# ======================
def perform_lda_topic_modeling(text_file, output_dir, n_topics=10):
    """执行主题建模并保存结果"""
    logging.info(f"\n{'=' * 50}\n进行LDA主题建模\n{'=' * 50}")
    logging.info(f"- 输入文件: {text_file}")
    logging.info(f"- 输出目录: {output_dir}")
    logging.info(f"- 主题数量: {n_topics}")

    os.makedirs(output_dir, exist_ok=True)
    lda_results_file = os.path.join(output_dir, "lda_results.npy")

    # 检查是否已经存在LDA结果
    if os.path.exists(lda_results_file):
        logging.info(f"已找到现有的LDA结果，直接加载: {lda_results_file}")
        lda_output = np.load(lda_results_file)
        lda_model = None  # 不需要重新训练模型
    else:
        # 读取文本数据
        with open(text_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f]

        # 基本文本预处理
        def preprocess(text):
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            return text

        texts = [preprocess(text) for text in texts]

        # 创建词袋模型
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        dtm = vectorizer.fit_transform(texts)

        # 训练LDA模型
        logging.info("\n训练LDA模型中...")
        lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            learning_method='online',
            random_state=42
        )
        lda_output = lda_model.fit_transform(dtm)

        # 保存结果
        np.save(lda_results_file, lda_output)
        logging.info(f"主题特征已保存到: {lda_results_file}")

    return lda_results_file, lda_model

# ======================
# 模型定义部分
# ======================
class EarlyFusionModel(nn.Module):
    """早期融合模型"""

    def __init__(self, topic_dim, use_bert=True):
        super().__init__()
        self.use_bert = use_bert
        self.text_dim = 768 if use_bert else 0

        if self.use_bert:
            self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.topic_proj = nn.Sequential(
            nn.Linear(topic_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        self.emotion_proj = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        self.fc = nn.Sequential(
            nn.Linear(self.text_dim + 128 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # 二分类
        )

    def forward(self, text_inputs=None, topic_feats=None, emotion_feats=None):
        # 文本特征
        text_feats = torch.zeros((topic_feats.size(0), self.text_dim),
                                 device=topic_feats.device)
        if self.use_bert and text_inputs:
            text_feats = self.bert(**text_inputs).last_hidden_state.mean(dim=1)

        # 主题特征
        topic_feats = self.topic_proj(topic_feats)

        # 情感特征
        emotion_feats = self.emotion_proj(emotion_feats)

        # 融合特征
        fused_feats = torch.cat([text_feats, topic_feats, emotion_feats], dim=1)
        return self.fc(fused_feats)

class AttentionFusionModel(nn.Module):
    """注意力融合模型"""

    def __init__(self, topic_dim, use_bert=True):
        super().__init__()
        self.use_bert = use_bert
        self.text_dim = 768 if use_bert else 0

        if self.use_bert:
            self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.topic_proj = nn.Sequential(
            nn.Linear(topic_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        self.emotion_proj = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=128, num_heads=4, batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(self.text_dim + 128 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # 二分类
        )

    def forward(self, text_inputs=None, topic_feats=None, emotion_feats=None):
        # 文本特征
        text_feats = torch.zeros((topic_feats.size(0), self.text_dim),
                                 device=topic_feats.device)
        if self.use_bert and text_inputs:
            text_feats = self.bert(**text_inputs).last_hidden_state.mean(dim=1)

        # 主题特征
        topic_feats = self.topic_proj(topic_feats)

        # 情感特征
        emotion_feats = self.emotion_proj(emotion_feats)

        # 跨模态注意力
        attn_output, _ = self.cross_attention(
            topic_feats.unsqueeze(1),
            emotion_feats.unsqueeze(1),
            emotion_feats.unsqueeze(1)
        )
        attn_output = topic_feats + attn_output.squeeze(1)

        # 融合特征
        fused_feats = torch.cat([text_feats, attn_output, emotion_feats], dim=1)
        return self.fc(fused_feats)

# ======================
# 数据加载部分
# ======================
class MultimodalDataset(Dataset):
    """多模态数据集类"""

    def __init__(self, text_file: str,
                 label_file: str,
                 sentiment_file: str,
                 use_lda=True,
                 lda_dir="lda_results",
                 use_bert=True):
        """
        参数:
            text_file: 文本文件路径
            label_file: 标签文件路径
            sentiment_file: 情感特征文件路径
            use_lda: 是否自动进行LDA主题建模
            lda_dir: LDA结果保存目录
            use_bert: 是否使用BERT特征
        """
        self.text_file = text_file
        self.label_file = label_file
        self.sentiment_file = sentiment_file
        self.use_lda = use_lda
        self.lda_dir = lda_dir
        self.use_bert = use_bert

        logging.info(f"\n{'=' * 50}\n加载数据集\n{'=' * 50}")
        logging.info(f"文本文件: {text_file}")
        logging.info(f"标签文件: {label_file}")
        logging.info(f"情感文件: {sentiment_file}")

        # 加载文本
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                self.texts = [line.strip() for line in f]
        except Exception as e:
            raise ValueError(f"加载文本文件失败: {e}")

        # 加载标签
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                self.labels = np.array([int(line.strip()) for line in f])
        except Exception as e:
            raise ValueError(f"加载标签文件失败: {e}")

        # 加载情感特征
        try:
            self.emotion_feats = np.loadtxt(sentiment_file)
        except Exception as e:
            raise ValueError(f"加载情感文件失败: {e}")

        # 处理主题特征
        self.topic_feats = None
        lda_file = os.path.join(lda_dir, "lda_results.npy")

        if use_lda and not os.path.exists(lda_file):
            logging.info(f"未找到现有的LDA结果，开始主题建模: {lda_file}")
            lda_file, _ = perform_lda_topic_modeling(text_file, lda_dir)

        try:
            self.topic_feats = np.load(lda_file)
            logging.info(f"加载主题文件: {lda_file}")
        except Exception as e:
            raise ValueError(f"加载主题文件失败: {e}")

        # 对齐数据长度
        min_length = min(len(self.texts), len(self.labels),
                         len(self.emotion_feats), len(self.topic_feats))
        logging.info(f"对齐后样本数: {min_length}")

        self.texts = self.texts[:min_length]
        self.labels = self.labels[:min_length]
        self.emotion_feats = self.emotion_feats[:min_length]
        self.topic_feats = self.topic_feats[:min_length]

        # 初始化BERT tokenizer
        if use_bert:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.max_seq_length = 128

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text_inputs = {}
        if self.use_bert:
            inputs = self.tokenizer(
                self.texts[idx],
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            text_inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        return (
            text_inputs,
            torch.tensor(self.topic_feats[idx], dtype=torch.float32),
            torch.tensor([self.emotion_feats[idx]], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

# ======================
# 模型评估函数
# ======================
def evaluate_model(model, val_loader, device):
    model.eval()
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            text_inputs, topic, emotion, labels = batch

            topic = topic.to(device)
            emotion = emotion.to(device)
            labels = labels.to(device)
            if 'use_bert' in locals() and use_bert:
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

            logits = model(
                text_inputs if 'use_bert' in locals() and use_bert else None,
                topic,
                emotion
            )

            preds = torch.argmax(logits, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    # 计算总体指标
    accuracy = accuracy_score(val_labels, val_preds)
    precision = precision_score(val_labels, val_preds)
    recall = recall_score(val_labels, val_preds)
    f1 = f1_score(val_labels, val_preds)
    conf_matrix = confusion_matrix(val_labels, val_preds)

    # 计算每个类别的指标
    fake_labels = [l for l in val_labels if l == 0]
    fake_preds = [p for i, p in enumerate(val_preds) if val_labels[i] == 0]
    real_labels = [l for l in val_labels if l == 1]
    real_preds = [p for i, p in enumerate(val_preds) if val_labels[i] == 1]

    fake_precision = precision_score(fake_labels, fake_preds) if fake_labels else 0.0
    fake_recall = recall_score(fake_labels, fake_preds) if fake_labels else 0.0
    fake_f1 = f1_score(fake_labels, fake_preds) if fake_labels else 0.0

    real_precision = precision_score(real_labels, real_preds) if real_labels else 0.0
    real_recall = recall_score(real_labels, real_preds) if real_labels else 0.0
    real_f1 = f1_score(real_labels, real_preds) if real_labels else 0.0

    logging.info("\n模型评估结果:")
    logging.info(f"准确率: {accuracy:.4f}")
    logging.info(f"精确率: {precision:.4f}")
    logging.info(f"召回率: {recall:.4f}")
    logging.info(f"F1值: {f1:.4f}")
    logging.info(f"混淆矩阵:\n{conf_matrix}")
    logging.info(f"假新闻精确率: {fake_precision:.4f}")
    logging.info(f"假新闻召回率: {fake_recall:.4f}")
    logging.info(f"假新闻F1值: {fake_f1:.4f}")
    logging.info(f"真新闻精确率: {real_precision:.4f}")
    logging.info(f"真新闻召回率: {real_recall:.4f}")
    logging.info(f"真新闻F1值: {real_f1:.4f}")

    return accuracy, precision, recall, f1, conf_matrix

# ======================
# 模型训练与评估
# ======================
def train_model(fusion_strategy,
                use_bert,
                batch_size,
                epochs,
                learning_rate,
                train_text_file,
                train_label_file,
                train_sentiment_file,
                val_text_file,
                val_label_file,
                val_sentiment_file,
                use_lda=True,
                lda_dir="lda_results"):
    """训练多模态分类模型"""
    logging.info("\n" + "=" * 50)
    logging.info("开始模型训练...")
    logging.info(f"- 融合策略: {fusion_strategy}")
    logging.info(f"- 使用BERT: {'是' if use_bert else '否'}")
    logging.info(f"- 批量大小: {batch_size}")
    logging.info(f"- 训练轮数: {epochs}")
    logging.info(f"- 学习率: {learning_rate}")
    logging.info(f"- 使用LDA主题建模: {'是' if use_lda else '否'}")
    logging.info(f"- LDA目录: {lda_dir}")
    logging.info("=" * 50)

    # 创建数据集
    logging.info("\n加载训练数据集...")
    train_dataset = MultimodalDataset(
        train_text_file,
        train_label_file,
        train_sentiment_file,
        use_lda=use_lda,
        lda_dir=lda_dir,
        use_bert=use_bert
    )

    logging.info("\n加载验证数据集...")
    val_dataset = MultimodalDataset(
        val_text_file,
        val_label_file,
        val_sentiment_file,
        use_lda=use_lda,
        lda_dir=lda_dir,
        use_bert=use_bert
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"\n使用设备: {device}")

    # 初始化模型
    logging.info("\n初始化模型...")
    n_topics = train_dataset.topic_feats.shape[1]

    if fusion_strategy == "early":
        model = EarlyFusionModel(n_topics, use_bert)
    else:
        model = AttentionFusionModel(n_topics, use_bert)

    model.to(device)
    logging.info(f"模型参数量: {sum(p.numel() for p in model.parameters())}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练循环
    logging.info("\n" + "=" * 50)
    logging.info(f"开始训练 {fusion_strategy} 融合模型")
    logging.info("=" * 50)

    best_val_acc = 0.0
    best_model_path = ""

    for epoch in range(1, epochs + 1):
        logging.info(f"\n轮次 {epoch}/{epochs}")
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        start_time = time.time()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
            text_inputs, topic, emotion, labels = batch

            # 移动到设备
            topic = topic.to(device)
            emotion = emotion.to(device)
            labels = labels.to(device)
            if use_bert:
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

            # 前向传播
            optimizer.zero_grad()
            logits = model(
                text_inputs if use_bert else None,
                topic,
                emotion
            )

            # 计算损失
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # 反向传播
            loss.backward()
            optimizer.step()

            # 计算准确率
            preds = torch.argmax(logits, dim=1)
            correct = (preds == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)

        # 计算训练指标
        train_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_samples
        epoch_time = time.time() - start_time

        # 验证评估
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                text_inputs, topic, emotion, labels = batch

                topic = topic.to(device)
                emotion = emotion.to(device)
                labels = labels.to(device)
                if use_bert:
                    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

                logits = model(
                    text_inputs if use_bert else None,
                    topic,
                    emotion
                )

                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # 计算验证指标
        val_acc = accuracy_score(val_labels, val_preds)
        fake_labels = [l for l in val_labels if l == 0]
        fake_preds = [p for i, p in enumerate(val_preds) if val_labels[i] == 0]
        real_labels = [l for l in val_labels if l == 1]
        real_preds = [p for i, p in enumerate(val_preds) if val_labels[i] == 1]

        fake_acc = accuracy_score(fake_labels, fake_preds) if fake_labels else 0.0
        real_acc = accuracy_score(real_labels, real_preds) if real_labels else 0.0

        # 更新最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_name = f"best_{fusion_strategy}_fusion_model.pth"
            torch.save(model.state_dict(), model_name)
            best_model_path = model_name
            logging.info(f"保存最佳模型: {model_name} (验证准确率: {val_acc:.4f})")

        # 打印结果
        logging.info(f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f} | 耗时: {epoch_time:.2f}s")
        logging.info(f"验证准确率: {val_acc:.4f}")
        logging.info(f"假新闻准确率: {fake_acc:.4f} ({len(fake_labels)}条)")
        logging.info(f"真新闻准确率: {real_acc:.4f} ({len(real_labels)}条)")

    logging.info(f"\n训练完成! 最佳模型保存至: {best_model_path} (最佳验证准确率: {best_val_acc:.4f})")

    # 验证集评估
    logging.info("\n开始验证集评估...")
    evaluate_model(model, val_loader, device)

    return best_model_path

# ======================
# 主执行函数
# ======================
def main():
    # 设置随机种子
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    logging.info("\n" + "=" * 50)
    logging.info("多模态假新闻检测系统")
    logging.info("=" * 50)

    # 获取原始数据路径
    print("\n请输入原始数据路径")
    train_file = input("训练集文件路径: ").strip()
    val_file = input("验证集文件路径: ").strip()
    output_dir = input("\n预处理数据输出目录: ").strip()
    lda_dir = input("LDA结果输出目录: ").strip()

    # 执行预处理
    try:
        data_dir, file_paths = preprocess_data(train_file, val_file, output_dir)
        logging.info("\n预处理完成! 开始训练模型...")
    except Exception as e:
        logging.error(f"\n预处理过程中出错: {str(e)}")
        logging.info("程序终止")
        return

    # 训练配置参数
    print("\n请选择融合策略:")
    fusion_strategy = input("输入 'early' 或 'attention': ").strip().lower()
    while fusion_strategy not in ['early', 'attention']:
        print("输入错误! 请重新选择融合策略")
        fusion_strategy = input("输入 'early' 或 'attention': ").strip().lower()

    use_bert = input("\n是否使用BERT文本编码器? (y/n): ").strip().lower() == 'y'
    use_lda = input("\n是否使用LDA主题建模? (y/n): ").strip().lower() == 'y'
    batch_size = int(input("\n批量大小(默认32): ") or 32)
    epochs = int(input("训练轮数(默认10): ") or 10)
    learning_rate = float(input("学习率(默认1e-5): ") or 1e-5)

    # 开始训练
    best_model = train_model(
        fusion_strategy=fusion_strategy,
        use_bert=use_bert,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        train_text_file=file_paths['train_text'],
        train_label_file=file_paths['train_labels'],
        train_sentiment_file=file_paths['train_sentiment'],
        val_text_file=file_paths['val_text'],
        val_label_file=file_paths['val_labels'],
        val_sentiment_file=file_paths['val_sentiment'],
        use_lda=use_lda,
        lda_dir=lda_dir
    )

    logging.info(f"\n训练完成! 最佳模型保存路径: {best_model}")

    # 显示LDA结果目录内容
    if use_lda and os.path.exists(lda_dir):
        logging.info("\nLDA主题建模结果:")
        for item in os.listdir(lda_dir):
            logging.info(f"- {item} ({os.path.getsize(os.path.join(lda_dir, item))} bytes)")
    elif use_lda:
        logging.info(f"\nLDA目录 {lda_dir} 不存在")

if __name__ == "__main__":
    main()