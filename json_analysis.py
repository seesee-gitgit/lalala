import json
import re


def parse_analysis_text(analysis_text):
    """从analysis字段的文本中提取标题、描述、实体和领域"""
    lines = [line.strip() for line in analysis_text.split('\n') if line.strip()]

    title = None
    description = None
    entities = None
    domain = None

    for line in lines:
        if line.startswith("标题:"):
            title = line.split(":", 1)[1].strip()
        elif line.startswith("描述:"):
            description = line.split(":", 1)[1].strip()
        elif line.startswith("实体:"):
            entities = line.split(":", 1)[1].strip()
        elif line.startswith("领域:") or line.startswith("所属领域:"):
            domain = line.split(":", 1)[1].strip()

    return {
        "标题": title,
        "描述": description,
        "实体": entities,
        "领域": domain
    }


# 读取并解析JSON文件
with open('topic_llm_analysis.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 遍历每个主题并输出信息
for idx, topic in enumerate(data, 1):
    print(f"主题ID: {topic['topic_id']}")
    print(f"关键词: {topic['keywords']}")

    # 解析analysis字段中的文本内容
    parsed = parse_analysis_text(topic['analysis'])

    print(f"标题: {parsed['标题']}")
    print(f"描述: {parsed['描述']}")
    print(f"实体: {parsed['实体']}")
    print(f"领域: {parsed['领域']}")
    print("-" * 80)