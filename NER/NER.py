import re
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_medical_entities_from_file(file_path):

    # 检查文件是否存在
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return None
    
    # 读取文件内容
    try:
        logger.info(f"正在读取文件: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        logger.info(f"文件读取成功，内容长度: {len(text)} 字符")
        
        # 提取实体
        entities = extract_medical_entities(text)
        return entities
        
    except UnicodeDecodeError:
        # 尝试其他编码
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                text = f.read()
            logger.info(f"使用GBK编码读取成功，内容长度: {len(text)} 字符")
            
            # 提取实体
            entities = extract_medical_entities(text)
            return entities
        except Exception as e:
            logger.error(f"读取文件失败: {e}")
            return None
    except Exception as e:
        logger.error(f"读取文件失败: {e}")
        return None

def extract_medical_entities(text):
    """从医疗文本中提取姓名、性别、生日和访视日期"""
    entities = {
        "姓名": None,
        "性别": None,
        "生日": None,
        "访视日期": None
    }
    
    # 姓名提取
    name_patterns = [
        r"姓名[:：\s]*([^\s，。,]+)",
        r"患者([^\s，。,]+)，",
        r"患者.*?([^\s，。,]+)(，|。).*?(男|女)",
        r"([^\s，。,]+)患者"
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, text)
        if match:
            entities["姓名"] = match.group(1).strip()
            break
    
    # 性别提取
    gender_patterns = [
        r"性别[:：\s]*(男|女)",
        r"患者.*?(男|女)性",
        r"(男|女)性患者"
    ]
    
    for pattern in gender_patterns:
        match = re.search(pattern, text)
        if match:
            # 确定捕获组索引
            group_index = 1
            if "性患者" in pattern:
                group_index = 1
            elif "患者.*?" in pattern:
                group_index = 1
                
            if group_index <= len(match.groups()):
                entities["性别"] = match.group(group_index).strip()
                break
    
    # 生日提取
    birth_patterns = [
        r"出生日期[:：\s]*(\d{4}[年/-]\d{1,2}[月/-]\d{1,2}日?)",
        r"生日[:：\s]*(\d{4}[年/-]\d{1,2}[月/-]\d{1,2}日?)",
        r"出生于[\s]*(\d{4}[年/-]\d{1,2}[月/-]\d{1,2}日?)",
        r"(\d{4}[年/-]\d{1,2}[月/-]\d{1,2}日?)出生"
    ]
    
    for pattern in birth_patterns:
        match = re.search(pattern, text)
        if match:
            entities["生日"] = match.group(1).strip()
            break
    
    # 访视日期提取
    visit_patterns = [
        r"就诊日期[:：\s]*(\d{4}[年/-]\d{1,2}[月/-]\d{1,2}日?)",
        r"访视日期[:：\s]*(\d{4}[年/-]\d{1,2}[月/-]\d{1,2}日?)",
        r"门诊日期[:：\s]*(\d{4}[年/-]\d{1,2}[月/-]\d{1,2}日?)",
        r"于(\d{4}[年/-]\d{1,2}[月/-]\d{1,2}日?)来我院",
        r"(\d{4}[年/-]\d{1,2}[月/-]\d{1,2}日?)就诊"
    ]
    
    for pattern in visit_patterns:
        match = re.search(pattern, text)
        if match:
            entities["访视日期"] = match.group(1).strip()
            break
    
    return entities

def process_directory(directory_path):

    results = []
    
    if not os.path.isdir(directory_path):
        logger.error(f"目录不存在: {directory_path}")
        return results
    
    files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
    logger.info(f"找到 {len(files)} 个txt文件")
    
    for file_name in files:
        file_path = os.path.join(directory_path, file_name)
        logger.info(f"处理文件: {file_name}")
        
        result = extract_medical_entities_from_file(file_path)
        if result:
            results.append({
                "file_name": file_name,
                "entities": result
            })
    
    return results

def export_results(results, output_file):

    import csv
    
    if not results:
        logger.warning("没有处理结果可导出")
        return
    
    try:
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            

            writer.writerow(['文件名', '姓名', '性别', '生日', '访视日期'])
            

            for result in results:
                writer.writerow([
                    result['file_name'],
                    result['entities']['姓名'] or '',
                    result['entities']['性别'] or '',
                    result['entities']['生日'] or '',
                    result['entities']['访视日期'] or ''
                ])
        
        logger.info(f"结果已导出到: {output_file}")
        
    except Exception as e:
        logger.error(f"导出结果失败: {e}")

# 使用示例
if __name__ == "__main__":
    # 单个文件处理
    file_path = "medical_record.txt"  # 替换为实际文件路径
    result = extract_medical_entities_from_file(file_path)
    if result:
        print(f"文件 {file_path} 的提取结果:")
        for key, value in result.items():
            print(f"  {key}: {value}")
 