import requests
import csv
from urllib.parse import quote

disease_list = [
    "Type 2 Diabetes",         # 替代 Type 2 Diabetes Mellitus
    "Hypertension",
    "Coronary Disease",        # 替代 Coronary Artery Disease
    "Stroke",                  # 替代 Cerebral Infarction
    "COPD",                    # 替代 Chronic Obstructive Pulmonary Disease
    "Cancer",                  # 替代 Breast Cancer（可以更细分如 Breast Neoplasms）
    "Osteoarthritis",
    "Depression",
    "Obesity"
]

base_url = "https://clinicaltrials.gov/api/v2/studies"
output_file = "icd_studies.csv"

all_rows = []

for disease in disease_list:
    print(f"正在提取 {disease} 数据...")
    encoded_disease = quote(disease)
    params = {
        "query.cond": encoded_disease,
        "pageSize": 100,
        "format": "json"
    }
    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            studies = data.get("studies", [])
            if not studies:
                print(f"{disease}: 没有找到研究。")
                continue
            for study in studies:
                protocol = study.get("protocolSection", {})
                name = protocol.get("identificationModule", {}).get("briefTitle", "")
                eligibility = protocol.get("eligibilityModule", {}).get("eligibilityCriteria", "")
                all_rows.append([disease, name.strip(), eligibility.strip()])
            print(f"{disease}: 提取成功，共 {len(studies)} 条研究。")
        else:
            print(f"{disease}: 请求失败，状态码：{response.status_code}")
    except Exception as e:
        print(f"{disease}: 提取过程中发生错误：{str(e)}")

# 写入CSV文件
with open(output_file, mode="w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Disease", "Study Name", "Eligibility Criteria"])
    writer.writerows(all_rows)

print(f"数据提取完毕，已保存为 {output_file}")

