import warnings
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os
import csv

# 忽略不必要的警告
warnings.filterwarnings("ignore")

# ICD编码到疾病名称映射
ICD_TO_DISEASE = {
    "E11": "Type 2 Diabetes",
    "I10": "Hypertension",
    "I25": "Coronary Disease",
    "I63": "Stroke",
    "J44": "COPD",
    "C50": "Cancer",
    "M19": "Osteoarthritis",
    "F32": "Depression",
    "E66": "Obesity"
}

class Retriever:
    def __init__(self, csv_path, embed_model_path="sentence-transformers/all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embed_model_path)
        self.df = pd.read_csv(csv_path)
        self.embeddings = self.embed_texts(self.df["Eligibility Criteria"].fillna("").tolist())

    def embed_texts(self, texts):
        return self.embedder.encode(texts, convert_to_tensor=False, normalize_embeddings=True)

    def retrieve(self, query, top_k=5):
        query_embedding = self.embed_texts([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_k_indices = similarities.argsort()[-top_k:][::-1]
        return self.df.iloc[top_k_indices][["Disease", "Study Name", "Eligibility Criteria"]]

    def query_by_icd(self, icd_code, top_k=10):
        disease = ICD_TO_DISEASE.get(icd_code.upper())
        if not disease:
            print(f"未知的ICD编码：{icd_code}")
            return pd.DataFrame()
        filtered_df = self.df[self.df["Disease"].str.lower() == disease.lower()]
        if filtered_df.empty:
            print(f"未在数据中找到与 {disease} 对应的研究。")
            return pd.DataFrame()
        return filtered_df.head(top_k)[["Disease", "Study Name", "Eligibility Criteria"]]

def build_prompt_for_entities(disease_name, docs):
    context = ""
    for idx, row in docs.iterrows():
        context += f"\n[研究项目: {row['Study Name']}]\n{row['Eligibility Criteria']}\n"

    prompt = f"""
你是一个医学信息抽取助手，请从以下{disease_name}相关研究中提取最常出现的纳入标准和排除标准中的实体（如年龄、疾病、指标等）及其取值范围。

{context}

请将结果以如下 JSON 格式输出：

{{
  "inclusion": [
    {{"attribute": "Age", "value": "≥ 18 years"}},
    {{"attribute": "NYHA Class", "value": "II or III"}},
    {{"attribute": "GFR", "value": "≥ 25 ml/min"}}
  ],
  "exclusion": [
    {{"attribute": "Myocardial infarction", "value": "within 3 months"}},
    {{"attribute": "Pulmonary artery aneurysm", "value": "present"}},
    {{"attribute": "CRT", "value": "within last 30 days"}}
  ]
}}
"""
    return prompt.strip()
# def build_prompt_for_entities(disease_name, docs):
#     context = ""
#     for idx, row in docs.iterrows():
#         context += f"\n[研究项目: {row['Study Name']}]\n{row['Eligibility Criteria']}\n"

#     prompt = f"""
# 你是一个医学信息抽取助手，任务是从以下与「{disease_name}」相关的临床研究中，提取出常见的纳入标准（inclusion criteria）与排除标准（exclusion criteria）。

# 请按以下要求进行抽取：

# 1. 每一条标准拆分为两个字段：
#    - attribute：表示临床属性、指标或事件的名称（如 Age, NYHA Class, GFR, Myocardial infarction 等）
#    - value：表示该属性的取值、范围、状态或时间条件（如 ≥ 18 years, II or III, ≥ 25 ml/min, within 3 months）

# 2. 如果标准包含具体数值、时间范围或等级，请将这些信息写入 value 中，不要写入 attribute。

# 3. 如果某条标准涉及多个医学事件（如“心梗、心绞痛或中风在近3个月内”），请拆分为多条记录，每条记录对应一个 attribute 和一个 value。

# 4. 避免使用模糊的 value，如 "yes"、"no"，应写出完整的判断标准（如 "present", "within 30 days", "≥ 1.5 mg/dL" 等）。

# 请将结果以如下 JSON 格式输出：

# {{
#   "inclusion": [
#     {{ "attribute": "Age", "value": "≥ 18 years" }},
#     {{ "attribute": "NYHA Class", "value": "II or III" }},
#     ...
#   ],
#   "exclusion": [
#     {{ "attribute": "Renal failure", "value": "Serum creatinine > 1.5 mg/dL" }},
#     {{ "attribute": "Myocardial infarction", "value": "within 3 months" }},
#     ...
#   ]
# }}

# 以下是研究资料：{context}
# """
#     return prompt.strip()


def extract_quadruples_from_json(answer_str, disease_name):
    """
    从大模型回答中提取四元组 (疾病名, 属性名, 属性值, 类型)
    """
    try:
        json_start = answer_str.find("{")
        json_data = json.loads(answer_str[json_start:])

        quadruples = []
        for section in ["inclusion", "exclusion"]:
            for item in json_data.get(section, []):
                if "attribute" in item and "value" in item:
                    quadruples.append((disease_name, item["attribute"], item["value"], section))
        return quadruples
    except Exception as e:
        print(f"解析失败: {e}")
        return []


def save_result_and_quadruples(disease_name, answer_text, quadruples, save_dir="outputs"):
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.join(save_dir, disease_name.replace(" ", "_"))

    # 保存 answer 原始输出
    with open(f"{base_name}_answer.txt", "w", encoding="utf-8") as f:
        f.write(answer_text)

    # 保存四元组 JSONL
    with open(f"{base_name}_quadruples.jsonl", "w", encoding="utf-8") as f:
        for quad in quadruples:
            json.dump(quad, f, ensure_ascii=False)
            f.write("\n")

    # 保存四元组 CSV
    with open(f"{base_name}_quadruples.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Disease", "Attribute", "Value", "Type"])
        writer.writerows(quadruples)

    print(f"\n已保存以下文件：\n- {base_name}_answer.txt\n- {base_name}_quadruples.jsonl\n- {base_name}_quadruples.csv")

# 加载大模型
print("正在加载大模型...")
tokenizer = AutoTokenizer.from_pretrained("./models/baichuan2-7b-chat", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "./models/baichuan2-7b-chat",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
model.eval()

# 初始化向量检索
retriever = Retriever("icd_studies.csv")

print("RAG系统准备就绪。请输入你的问题（支持 ICD 编码）")
while True:
    query = input("\n问题或ICD编码: ")
    if query.lower() in ["exit", "quit", "q"]:
        break

    if query.upper() in ICD_TO_DISEASE:
        disease = ICD_TO_DISEASE[query.upper()]
        docs = retriever.query_by_icd(query.upper(), top_k=5)
        if docs.empty:
            print("未找到与此 ICD 编码相关的研究。")
            continue
        prompt = build_prompt_for_entities(disease, docs)
    else:
        docs = retriever.retrieve(query, top_k=5)
        disease = query 
        prompt = f"请根据以下Eligibility Criteria回答问题：\n{docs.to_string(index=False)}\n\n问题：{query}"

    messages = [{"role": "user", "content": prompt}]
    answer = model.chat(tokenizer, messages)
    print("\n 提取结果：")
    print(answer)

    quads = extract_quadruples_from_json(answer, disease)
    for q in quads:
        print(q)


    save_result_and_quadruples(disease, answer, quads)
