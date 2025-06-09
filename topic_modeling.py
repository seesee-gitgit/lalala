import os
import re
import json
import logging
import random
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from wordcloud import WordCloud
import seaborn as sns
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim import corpora, models
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary, MmCorpus

# 配置日志
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Ollama API服务地址
OLLAMA_BASE_URL = "http://localhost:11434/api"


class LDAProcessor:
    def __init__(self, input_file, output_dir):
        self.input_file = input_file
        self.output_dir = output_dir
        self.num_samples = 10  # 固定样本数量为10个
        self.original_docs = []
        self.processed_docs = []
        self.lda_model = None
        self.dictionary = None
        self.corpus = None

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 下载必要的nltk数据
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)

    def load_and_sample_data(self):
        """加载数据并随机抽样，专门提取post_text列"""
        try:
            # 读取文件并检测分隔符
            with open(self.input_file, 'r', encoding='utf-8', errors='replace') as f:
                first_line = f.readline().strip()

            # 检测分隔符（制表符或逗号）
            sep = '\t' if '\t' in first_line else ','

            # 读取整个文件
            df = pd.read_csv(self.input_file, sep=sep, encoding='utf-8', on_bad_lines='warn')

            # 检查是否存在post_text列
            if 'post_text' not in df.columns:
                logger.error("文件中未找到'post_text'列！请确保文件包含文本数据列")
                return False

            documents = df['post_text'].astype(str).tolist()

            if len(documents) < self.num_samples:
                logger.warning(f"文件只有{len(documents)}条数据，使用全部数据")
                self.original_docs = documents
            else:
                self.original_docs = random.sample(documents, self.num_samples)

            logger.info(f"成功加载{len(self.original_docs)}条文档")
            return True
        except Exception as e:
            logger.error(f"加载数据失败: {str(e)}")
            return False

    def preprocess_text(self, text):
        """预处理单条文本"""
        # 移除URL和HTML标签
        text = re.sub(r'http\S+|www\S+|<.*?>', '', text)
        # 移除非字母字符，保留空格
        text = re.sub(r'[^a-zA-Z\s]', '', text).lower()

        # 分词
        tokens = word_tokenize(text)

        # 去停用词和短词
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]

        # 词形还原
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        return tokens

    def preprocess_data(self):
        """预处理所有文档"""
        self.processed_docs = [self.preprocess_text(doc) for doc in self.original_docs]
        logger.info("文本预处理完成")
        return True

    def train_lda_model(self, num_topics=5, passes=15):
        """训练LDA模型"""
        try:
            # 构建词典
            self.dictionary = Dictionary(self.processed_docs)
            self.dictionary.filter_extremes(no_below=1, no_above=0.8)
            logger.info(f"词典大小: {len(self.dictionary)} 个唯一词")

            # 创建语料库
            self.corpus = [self.dictionary.doc2bow(doc) for doc in self.processed_docs]

            # 动态调整主题数量
            num_topics = min(len(self.corpus), max(2, num_topics))

            # 训练LDA模型
            self.lda_model = LdaModel(
                corpus=self.corpus,
                id2word=self.dictionary,
                num_topics=num_topics,
                passes=passes,
                alpha='auto',
                eta='auto',
                random_state=42
            )

            logger.info(f"LDA模型训练完成，主题数: {num_topics}")
            return True
        except Exception as e:
            logger.error(f"训练LDA模型失败: {str(e)}")
            return False

    def generate_pyldavis(self):
        """生成pyLDAvis交互式主题可视化"""
        try:
            vis_data = gensimvis.prepare(self.lda_model, self.corpus, self.dictionary, sort_topics=False)
            output_file = os.path.join(self.output_dir, 'topic_visualization.html')
            pyLDAvis.save_html(vis_data, output_file)
            logger.info(f"pyLDAvis交互图已保存: {output_file}")
            return True
        except Exception as e:
            logger.error(f"生成pyLDAvis失败: {str(e)}")
            return False

    def generate_wordclouds(self):
        """为每个主题生成词云图"""
        try:
            topics = self.lda_model.show_topics(num_topics=-1, num_words=20, formatted=False)
            wordcloud_dir = os.path.join(self.output_dir, 'wordclouds')
            os.makedirs(wordcloud_dir, exist_ok=True)

            for topic_id, word_weights in topics:
                word_freq = {word: weight for word, weight in word_weights}
                wc = WordCloud(width=800, height=600, background_color='white',
                               max_words=100, colormap='viridis').generate_from_frequencies(word_freq)

                plt.figure(figsize=(10, 8))
                plt.imshow(wc, interpolation='bilinear')
                plt.title(f'主题 #{topic_id + 1} 关键词词云', fontsize=16)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(wordcloud_dir, f'topic_{topic_id + 1}_wordcloud.png'))
                plt.close()

            logger.info(f"词云图已保存到: {wordcloud_dir}")
            return True
        except Exception as e:
            logger.error(f"生成词云图失败: {str(e)}")
            return False

    def generate_heatmap(self, max_docs=50):
        """生成文档-主题分布热力图"""
        try:
            # ======== 添加中文支持 ========
            import platform
            system = platform.system()
            if system == 'Windows':
                plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统使用黑体
            elif system == 'Darwin':  # macOS
                plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS使用Arial Unicode MS
            else:  # Linux
                plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  # Linux使用文泉驿微米黑

            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            # =============================

            doc_topics = []
            for doc in self.corpus:
                topics_probs = self.lda_model.get_document_topics(doc)
                probs = np.zeros(self.lda_model.num_topics)
                for topic_id, prob in topics_probs:
                    probs[topic_id] = prob
                doc_topics.append(probs)

            doc_topics_df = pd.DataFrame(
                doc_topics,
                columns=[f'主题 {i + 1}' for i in range(self.lda_model.num_topics)]
            )
            doc_topics_df.insert(0, '文档ID', range(len(doc_topics_df)))

            if self.original_docs:
                doc_summaries = [doc[:50] + '...' if len(doc) > 50 else doc for doc in self.original_docs]
                doc_topics_df['文档摘要'] = doc_summaries

            if len(doc_topics_df) > max_docs:
                doc_topics_df = doc_topics_df.sample(max_docs)

            plt.figure(figsize=(12, min(15, len(doc_topics_df) // 2)))
            heatmap_df = doc_topics_df.set_index('文档ID').drop('文档摘要', axis=1, errors='ignore')

            cmap = plt.cm.get_cmap('viridis').copy()
            cmap.set_under('white')

            sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap=cmap, vmin=0.01,
                        cbar_kws={'label': '主题概率'})
            plt.title('文档-主题分布热力图', fontsize=16)
            plt.tight_layout()

            heatmap_path = os.path.join(self.output_dir, 'document_topic_heatmap.png')
            plt.savefig(heatmap_path)
            plt.close()

            logger.info(f"热力图已保存: {heatmap_path}")
            return True
        except Exception as e:
            logger.error(f"生成热力图失败: {str(e)}")
            return False

    def generate_text(self, prompt, model="deepseek-r1:1.5b", temperature=0.7, max_tokens=1024):
        """调用Ollama API生成文本"""
        url = f"{OLLAMA_BASE_URL}/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": False
        }

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            logger.error(f"API请求错误: {str(e)}")
            return ""

    def analyze_with_llm(self, model_name="deepseek-r1:1.5b"):
        """使用大语言模型分析各主题内容"""
        try:
            topics = self.lda_model.show_topics(num_topics=-1, num_words=30, formatted=False)
            topic_analyses = []

            for topic_id, word_weights in topics:
                keywords = ", ".join([f"{word}({weight:.3f})" for word, weight in word_weights])

                prompt = f"""
                你是一名数据分析师，需要根据LDA主题模型的关键词描述分析主题内容。
                主题的关键词及其权重如下：
                {keywords}

                请完成以下任务：
                1. 为这个主题创建一个简短标题（10字以内）
                2. 用一段话（100字以内）描述主题的核心内容
                3. 从关键词中识别出3个最有代表性的实体
                4. 推测这个主题可能属于什么领域或学科

                请按以下格式输出：
                标题: [主题标题]
                描述: [主题描述]
                实体: [实体1, 实体2, 实体3]
                领域: [所属领域]
                """

                analysis = self.generate_text(prompt, model_name)
                topic_analyses.append({
                    "topic_id": topic_id + 1,
                    "keywords": keywords,
                    "analysis": analysis
                })

            # 保存分析结果
            json_path = os.path.join(output_dir, 'topic_llm_analysis.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(topic_analyses, f, ensure_ascii=False, indent=2)

            logger.info(f"LLM主题分析已保存: {json_path}")
            return True
        except Exception as e:
            logger.error(f"LLM分析失败: {str(e)}")
            return False

    def run_analysis(self, use_llm=True):
        """执行完整的分析流程"""
        logger.info("开始LDA主题建模与分析")

        if not self.load_and_sample_data():
            return False
        if not self.preprocess_data():
            return False
        if not self.train_lda_model():
            return False

        # 可视化分析
        self.generate_pyldavis()
        self.generate_wordclouds()
        self.generate_heatmap()

        # LLM主题分析
        if use_llm:
            self.analyze_with_llm()

        logger.info(f"分析完成! 结果保存在: {self.output_dir}")
        return True


def get_user_input():
    """获取用户输入的文件路径"""
    # 输入文件
    input_file = input("请输入数据文件路径: ")
    while not os.path.exists(input_file):
        print(f"错误: 文件 '{input_file}' 不存在!")
        input_file = input("请重新输入数据文件路径: ")

    # 输出目录
    output_dir = input("请输入结果输出目录: ").strip()
    if not output_dir:
        # 默认输出目录：当前目录下的results文件夹
        output_dir = os.path.join(os.getcwd(), "lda_results")
        print(f"使用默认输出目录: {output_dir}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    return input_file, output_dir


if __name__ == "__main__":
    print("=" * 60)
    print("LDA主题建模与分析工具")
    print("=" * 60)
    print("请按照提示输入以下信息:")

    # 获取用户输入
    input_file, output_dir = get_user_input()

    # 创建处理器并执行分析
    processor = LDAProcessor(input_file, output_dir)

    # 询问是否使用LLM分析
    use_llm = input("是否使用大语言模型分析主题内容? (y/n, 默认y): ").strip().lower()
    use_llm = use_llm != 'n'

    # 执行分析
    processor.run_analysis(use_llm=use_llm)

    print("\n" + "=" * 60)
    print(f"分析完成! 结果文件已保存到: {output_dir}")
    print("-" * 60)
    print("包含以下文件:")
    print(f"  - 交互式主题可视化: {os.path.join(output_dir, 'topic_visualization.html')}")
    print(f"  - 主题词云图: {os.path.join(output_dir, 'wordclouds')} 目录")
    print(f"  - 文档-主题热力图: {os.path.join(output_dir, 'document_topic_heatmap.png')}")
    if use_llm:
        print(f"  - LLM主题分析结果: {os.path.join(output_dir, 'topic_llm_analysis.json')}")
    print("=" * 60)
    print("\n提示: 打开topic_visualization.html文件查看交互式主题模型")