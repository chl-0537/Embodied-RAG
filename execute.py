import os
import docx
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import re
from embedding import get_azure_embedding  # 导入embedding函数
import time
from sentence_transformers import CrossEncoder
from itertools import chain
import numpy as np
import json
 
# Azure OpenAI 配置
key = "bc709d6234e04a80ab2d744eb2434086"
os.environ["AZURE_OPENAI_API_KEY"] = key
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://lechuang.openai.azure.com/"
os.environ["AZURE_OPENAI_API_VERSION"] = "2023-05-15"

# DeepSeek 配置
base_url = "https://api.deepseek.com"
base_model = "deepseek-chat"
api_key = "sk-b1882ddd341f4871bbcf093efe68ddf5"  # 替换为您的API密钥
client = openai.OpenAI(api_key=api_key, base_url=base_url)

class EnhancedRuleProcessor:
    def __init__(self, rules_files, load_existing_db=True):
        """初始化增强版规则处理器"""
        self.rules_files = rules_files
        self.embeddings = get_azure_embedding()
        self.rules_db = None
        
        # 初始化交叉编码器用于重排序
        self.cross_encoder = CrossEncoder('BAAI/bge-reranker-base', device='cpu')
        
        # 加载或创建向量数据库
        if load_existing_db and os.path.exists("faiss_index"):
            print("加载已存在的向量数据库...")
            self.rules_db = FAISS.load_local("faiss_index", self.embeddings, allow_dangerous_deserialization=True)
            print("向量数据库加载完成")
        else:
            print("创建新的向量数据库...")
            self.initialize_vector_db()

    def parse_rule_file(self, file_path):
        """解析规则文件，提取批注、上下文，并用模型生成总结"""
        rules = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                sections = content.split('--------------------------------------------------')
                for section in sections:
                    if not section.strip():
                        continue
                    # 提取批注和上下文
                    comment_match = re.search(r'批注:(.*?)(?=上下文内容:|$)', section, re.DOTALL)
                    context_match = re.search(r'上下文内容:(.*?)(?=$)', section, re.DOTALL)
                    if comment_match and context_match:
                        comment = comment_match.group(1).strip()
                        context = context_match.group(1).strip()
                        # 用大模型生成总结
                        summary = self.generate_summary(comment, context)
                        rules.append({
                            'comment': comment,
                            'context': context,
                            'summary': summary
                        })
            print(f"从 {file_path} 成功解析并生成 {len(rules)} 条规则")
            return rules
        except Exception as e:
            print(f"解析文件 {file_path} 时出错: {str(e)}")
            return []

    def generate_summary(self, comment, context):
        """用大模型生成规则总结"""
        prompt = f"""请根据以下批注和上下文内容，总结一条简明的规则。
要求：
1. 保持原意，不要添加或删减关键信息
2. 使用规范的法律/招标文件用语
3. 确保总结的规则具有可操作性
4. 如果批注中已有明确的规则表述，优先使用原文
5. 注意保护商业机密，避免要求公开敏感信息
6. 检测报告要求需明确对应标准

批注：{comment}
上下文内容：{context}

请按以下格式输出：
规则总结：[规则内容]
规则类型：[法律要求/技术规范/程序要求/其他]
关键点：[列出规则中的关键要素]
"""
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                timeout=60
            )
            summary = response.choices[0].message.content.strip()
            print(f"生成规则总结: {summary[:50]}...")
            return summary
        except Exception as e:
            print(f"生成规则总结时出错: {str(e)}")
            return "（模型生成总结失败）"

    def verify_summary(self, comment, context, summary):
        verify_prompt = f"""请验证以下规则总结是否准确反映了原始批注和上下文的内容。
如果发现不准确，请指出具体问题。

原始批注：{comment}
原始上下文：{context}
规则总结：{summary}

请按以下格式输出：
1. 准确性评分（1-10分）：
2. 是否存在遗漏：
3. 是否存在错误：
4. 是否需要修改：
5. 修改建议（如有）：
"""
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": verify_prompt}],
                temperature=0.1
            )
            verification = response.choices[0].message.content
            
            # 解析验证结果
            if "准确性评分" in verification:
                score = int(re.search(r"准确性评分（1-10分）：(\d+)", verification).group(1))
                if score < 8:
                    # 如果评分低于8分，重新生成总结
                    return self.generate_summary(comment, context)
            return summary
        except Exception as e:
            print(f"验证规则总结时出错: {str(e)}")
            return summary

    def generate_rule_with_verification(self, comment, context):
        # 第一次生成
        initial_summary = self.generate_summary(comment, context)
        
        # 验证
        verified_summary = self.verify_summary(comment, context, initial_summary)
        
        # 如果验证发现问题，进行修正
        if verified_summary != initial_summary:
            print("规则总结需要修正，进行重新生成...")
            final_summary = self.generate_summary(comment, context)
        else:
            final_summary = verified_summary
        
        return {
            'comment': comment,
            'context': context,
            'summary': final_summary,
            'verification_status': 'verified'
        }

    def initialize_vector_db(self):
        """初始化向量数据库"""
        try:
            all_rules = []
            all_metadata = []
            
            # 创建规则备份文件
            backup_file = "rules_backup.txt"
            # csv_file = "rules_backup.csv"
            
            # # 创建并写入CSV文件头
            # with open(csv_file, 'w', encoding='utf-8', newline='') as csv:
            #     csv.write("批注,上下文,规则总结\n")
            
            # 创建TXT备份文件
            with open(backup_file, 'w', encoding='utf-8') as backup:
                backup.write("向量数据库规则备份\n")
                backup.write("生成时间: " + time.strftime('%Y-%m-%d %H:%M:%S') + "\n")
                backup.write("=" * 50 + "\n\n")
            
            # 收集所有规则
            for file_path in self.rules_files:
                if not os.path.exists(file_path):
                    print(f"文件不存在: {file_path}")
                    continue
                    
                rules = self.parse_rule_file(file_path)
                for rule in rules:
                    rule_text = f"批注: {rule['comment']}\n上下文: {rule['context']}\n总结: {rule['summary']}"
                    all_rules.append(rule_text)
                    all_metadata.append(rule)
                    
                    # 将规则文本写入TXT备份文件
                    with open(backup_file, 'a', encoding='utf-8') as backup:
                        backup.write(f"总结: {rule['summary']}" + "\n")
                        backup.write("-" * 50 + "\n\n")
                    
                    # # 将规则写入CSV文件
                    # with open(csv_file, 'a', encoding='utf-8', newline='') as csv:
                    #     # 处理可能包含逗号的内容
                    #     comment = rule['comment'].replace('"', '""')  # 处理引号
                    #     context = rule['context'].replace('"', '""')
                    #     summary = rule['summary'].replace('"', '""')
                        
                    #     # 如果内容包含逗号，用引号包裹
                    #     if ',' in comment or ',' in context or ',' in summary:
                    #         csv.write(f'"{comment}","{context}","{summary}"\n')
                    #     else:
                    #         csv.write(f'{comment},{context},{summary}\n')
            
            if not all_rules:
                raise ValueError("没有找到有效的规则")
            
            print(f"总共收集到 {len(all_rules)} 条规则")
            print(f"规则已备份到文件: {backup_file}")
            print(f"规则已备份到CSV文件: {csv_file}")
            
            # 分批处理规则
            batch_size = 20  # 每批处理的规则数量
            total_batches = (len(all_rules) + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                while True:
                    try:
                        start_idx = batch_idx * batch_size
                        end_idx = min((batch_idx + 1) * batch_size, len(all_rules))
                        
                        print(f"处理第 {batch_idx + 1}/{total_batches} 批 (规则 {start_idx + 1} 到 {end_idx})")
                        
                        batch_texts = all_rules[start_idx:end_idx]
                        batch_metadata = all_metadata[start_idx:end_idx]
                        
                        # 创建或更新向量数据库
                        if batch_idx == 0:
                            self.rules_db = FAISS.from_texts(
                                texts=batch_texts,
                                embedding=self.embeddings,
                                metadatas=batch_metadata
                            )
                        else:
                            # 添加到现有的向量数据库
                            self.rules_db.add_texts(
                                texts=batch_texts,
                                metadatas=batch_metadata
                            )
                        
                        print(f"成功处理第 {batch_idx + 1} 批")
                        # 成功处理后等待一小段时间
                        time.sleep(1)
                        break  # 如果成功，跳出重试循环
                        
                    except Exception as e:
                        if "429" in str(e):
                            wait_time = 65  # 等待65秒
                            print(f"达到API限制，等待 {wait_time} 秒后重试...")
                            time.sleep(wait_time)
                            continue
                        else:
                            raise  # 如果是其他错误，直接抛出
            
            print(f"成功加载所有规则到向量数据库")
            
            # 保存向量数据库到本地
            self.rules_db.save_local("faiss_index")
            print("向量数据库已保存到本地")
            
        except Exception as e:
            print(f"初始化向量数据库时出错: {str(e)}")
            raise

    def get_relevant_rules(self, query, top_k=5):
        """多路检索策略获取相关规则"""
        results = []
        
        try:
            # 策略1：直接相似度搜索
            similar_rules = self.rules_db.similarity_search_with_score(query, k=top_k)
            results.extend([(rule, score, 'similarity') for rule, score in similar_rules])
            
            # 策略2：关键词提取搜索
            keywords = self._extract_keywords(query)
            for keyword in keywords:
                keyword_rules = self.rules_db.similarity_search_with_score(keyword, k=3)
                results.extend([(rule, score, 'keyword') for rule, score in keyword_rules])
            
            # 策略3：分段搜索
            segments = self._split_query(query)
            for segment in segments:
                segment_rules = self.rules_db.similarity_search_with_score(segment, k=2)
                results.extend([(rule, score, 'segment') for rule, score in segment_rules])
            
            # 去重
            unique_rules = {}
            for rule, score, strategy in results:
                rule_id = rule.metadata['comment']  # 使用批注作为唯一标识
                if rule_id not in unique_rules or score < unique_rules[rule_id][1]:
                    unique_rules[rule_id] = (rule, score, strategy)
            
            # 重排序
            reranked_rules = self._rerank_rules(query, list(unique_rules.values()))
            
            return reranked_rules
            
        except Exception as e:
            print(f"规则检索出错: {str(e)}")
            return []

    def _extract_keywords(self, text):
        """提取关键词"""
        # 这里可以使用更复杂的关键词提取算法
        words = text.split()
        return [w for w in words if len(w) > 1]

    def _split_query(self, query):
        """将查询分段"""
        # 简单的按句子分割
        segments = [s.strip() for s in query.split('。') if s.strip()]
        return segments

    def _rerank_rules(self, query, rules):
        """使用交叉编码器重排序规则"""
        try:
            # 准备交叉编码器的输入
            pairs = []
            for rule, _, _ in rules:
                rule_text = f"批注: {rule.metadata['comment']}\n上下文: {rule.metadata['context']}\n总结: {rule.metadata['summary']}"
                pairs.append([query, rule_text])
            
            # 计算相关性分数
            scores = self.cross_encoder.predict(pairs)
            
            # 结合原始分数和重排序分数
            reranked = []
            for i, (rule, orig_score, strategy) in enumerate(rules):
                rerank_score = float(scores[i])
                # 综合分数 = 0.7 * 重排序分数 + 0.3 * 原始分数
                final_score = 0.7 * rerank_score + 0.3 * (1 - orig_score)
                reranked.append((rule, final_score, strategy))
            
            # 按最终分数排序
            reranked.sort(key=lambda x: x[1], reverse=True)
            
            return reranked
            
        except Exception as e:
            print(f"重排序出错: {str(e)}")
            return rules

    def categorize_rule(self, summary):
        category_prompt = f"""请对以下规则进行分类和标注：

规则内容：{summary}

请按以下格式输出：
1. 规则类型：[法律要求/技术规范/程序要求/其他]
2. 适用场景：[列出适用的具体场景]
3. 关键要素：[列出规则中的关键要素]
4. 相关法规：[列出相关的法规或标准]
5. 保密要求：[说明需要保护的商业机密]
6. 检测要求：[如涉及检测，列出具体标准和要求]
"""
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": category_prompt}],
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"规则分类时出错: {str(e)}")
            return None

    def process_rule(self, comment, context):
        try:
            # 1. 生成初始规则总结
            initial_summary = self.generate_summary(comment, context)
            
            # 2. 验证规则总结
            verified_summary = self.verify_summary(comment, context, initial_summary)
            
            # 3. 验证检测报告要求（如果涉及）
            if "检测报告" in verified_summary:
                detection_verification = self.verify_detection_report(verified_summary)
                if detection_verification:
                    # 根据验证结果调整规则
                    verified_summary = self.adjust_detection_requirements(verified_summary, detection_verification)
            
            # 4. 分类和标注
            categorization = self.categorize_rule(verified_summary)
            
            # 5. 构建完整的规则对象
            rule = {
                'comment': comment,
                'context': context,
                'summary': verified_summary,
                'categorization': categorization,
                'verification_status': 'verified',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # 6. 记录处理日志
            self.log_rule_processing(rule)
            
            return rule
            
        except Exception as e:
            print(f"规则处理出错: {str(e)}")
            return None

    def log_rule_processing(self, rule):
        log_entry = {
            'timestamp': rule['timestamp'],
            'rule_id': hash(rule['summary']),  # 使用规则内容的哈希值作为ID
            'verification_status': rule['verification_status'],
            'categorization': rule['categorization']
        }
        
        # 保存到日志文件
        with open('rule_processing_log.json', 'a', encoding='utf-8') as f:
            json.dump(log_entry, f, ensure_ascii=False)
            f.write('\n')

    def verify_detection_report(self, summary):
        verify_prompt = f"""请验证以下规则中关于检测报告的要求是否完整：

规则内容：{summary}

请检查：
1. 是否明确指定了检测标准
2. 是否说明了CMA/CNAS标识要求
3. 是否避免了要求公开商业机密
4. 是否明确了检测报告的有效期
5. 是否说明了检测机构的资质要求

请按以下格式输出：
1. 完整性评分（1-10分）：
2. 缺失要素：
3. 需要补充的内容：
4. 需要调整的内容：
"""
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": verify_prompt}],
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"验证检测报告要求时出错: {str(e)}")
            return None

    def adjust_detection_requirements(self, summary, verification):
        adjust_prompt = f"""请根据以下验证结果，调整检测报告要求：

原始规则：{summary}
验证结果：{verification}

请按以下原则调整：
1. 明确指定检测标准
2. 说明CMA/CNAS标识要求
3. 避免要求公开商业机密
4. 明确检测报告有效期
5. 说明检测机构资质要求

请输出调整后的完整规则：
"""
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": adjust_prompt}],
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"调整检测报告要求时出错: {str(e)}")
            return summary

def extract_docx_content(doc_path):
    """提取Word文档内容"""
    try:
        doc = docx.Document(doc_path)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # 提取段落内容
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text)
                
        # 提取表格内容
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        full_text.append(cell.text)
        
        # 分割文本
        return text_splitter.split_text("\n".join(full_text))
    except Exception as e:
        print(f"文档处理错误: {str(e)}")
        return []

def check_content(rule_processor, content_chunk):
    """增强版内容检查"""
    try:
        # 获取相关规则
        relevant_rules = rule_processor.get_relevant_rules(content_chunk)
        
        if not relevant_rules:
            return None
        
        # 降低相关性阈值，获取更多潜在规则
        filtered_rules = [rule for rule, score, strategy in relevant_rules if score > 0.4][:5]
        
        if not filtered_rules:
            return None
        
        # 构建更严格的提示词
        rules_text = "\n\n".join([
            f"规则 {i+1}:\n"
            f"批注: {rule.metadata['comment']}\n"
            f"上下文: {rule.metadata['context']}\n"
            f"总结: {rule.metadata['summary']}"
            for i, rule in enumerate(filtered_rules)
        ])
        
        prompt = f"""
        请对以下招标文件内容进行严格审查，找出所有可能存在的问题。审查要求：

        1. 全面性要求：
           - 检查每个条款的完整性
           - 检查条款之间的关联性
           - 检查是否存在遗漏的必要内容
           - 检查是否存在冗余或重复内容

        2. 规范性要求：
           - 检查是否符合法律法规
           - 检查是否符合行业标准
           - 检查是否符合招标文件编制规范
           - 检查是否存在歧视性条款

        3. 明确性要求：
           - 检查条款表述是否清晰
           - 检查是否存在歧义
           - 检查是否缺少必要的量化指标
           - 检查是否缺少必要的时限要求

        4. 可操作性要求：
           - 检查条款是否具有可执行性
           - 检查是否缺少必要的程序性规定
           - 检查是否缺少必要的保障措施
           - 检查是否缺少必要的验收标准

        5. 风险控制要求：
           - 检查是否存在潜在风险
           - 检查是否缺少必要的风险控制措施
           - 检查是否缺少必要的违约责任
           - 检查是否缺少必要的争议解决机制

        规则：
        {rules_text}
        
        待审查内容：
        {content_chunk}
        
        请按以下格式输出：
        1. 问题描述：
           - 具体问题
           - 问题类型（严重/一般/轻微）
           - 问题影响（对招标/投标/评标/合同执行的影响）
        
        2. 违反规则：
           - 具体违反的规则
           - 违反程度
           - 违反原因
        
        3. 修改建议：
           - 具体修改方案
           - 修改理由
           - 修改后的预期效果
        
        4. 其他建议：
           - 补充建议
           - 优化建议
           - 风险提示
        
        如果发现任何问题，无论大小，都需要详细说明。如果内容完全符合规则，请说明符合的具体表现。
        """
        
        # 调用API进行分析
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        result = response.choices[0].message.content
        if "完全符合规则" not in result:
            return {
                "content": content_chunk,
                "analysis": result,
                "relevant_rules": rules_text,
                "match_scores": [(rule.metadata['comment'], score, strategy) 
                               for rule, score, strategy in relevant_rules[:5]]
            }
        return None
        
    except Exception as e:
        print(f"内容检查错误: {str(e)}")
        return None

def save_report(results, output_path):
    """增强版报告保存"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("招标文件审查报告\n")
        f.write("=" * 50 + "\n\n")
        
        if not results:
            f.write("未发现违规内容。\n")
            return
        
        # 添加详细统计信息
        f.write("审查统计信息：\n")
        f.write(f"总问题数：{len(results)}\n")
        
        # 按问题类型统计
        problem_types = {
            '严重': 0,
            '一般': 0,
            '轻微': 0
        }
        
        # 按问题领域统计
        problem_areas = {
            '全面性': 0,
            '规范性': 0,
            '明确性': 0,
            '可操作性': 0,
            '风险控制': 0
        }
        
        for result in results:
            analysis = result['analysis']
            # 统计问题类型
            if '严重' in analysis:
                problem_types['严重'] += 1
            elif '一般' in analysis:
                problem_types['一般'] += 1
            elif '轻微' in analysis:
                problem_types['轻微'] += 1
                
            # 统计问题领域
            for area in problem_areas.keys():
                if area in analysis:
                    problem_areas[area] += 1
        
        f.write("\n问题类型分布：\n")
        for type_name, count in problem_types.items():
            f.write(f"- {type_name}问题：{count}个\n")
            
        f.write("\n问题领域分布：\n")
        for area, count in problem_areas.items():
            f.write(f"- {area}问题：{count}个\n")
            
        f.write("\n" + "=" * 50 + "\n\n")
        
        # 详细问题列表
        for i, result in enumerate(results, 1):
            f.write(f"问题 {i}\n")
            f.write("-" * 30 + "\n")
            f.write(f"相关内容：\n{result['content']}\n\n")
            f.write("匹配规则及相关度：\n")
            for rule, score, strategy in result['match_scores']:
                f.write(f"- {rule} (相关度: {score:.2f}, 策略: {strategy})\n")
            f.write(f"\n规则详情：\n{result['relevant_rules']}\n\n")
            f.write(f"分析结果：\n{result['analysis']}\n\n")
            f.write("=" * 50 + "\n\n")

def main():
    try:
        # 规则文件列表
        rules_files = [
            "./result/工程类招标文件初稿.txt",
            "./result/成品软件类招标文件初稿1.txt",
            "./result/服务类项目招标文件初稿1.txt",
            "./result/服务类项目招标文件初稿2.txt",
            "./result/物业类项目招标文件初稿.txt",
            "./result/设备类招标文件初稿1.txt",
            "./result/设备类招标文件初稿2.txt"
        ]
        
        print("开始初始化规则处理器...")
        rule_processor = EnhancedRuleProcessor(rules_files)
        
        # 处理招标文件列表
        docx_path_files = [
            # "./file/工程类招标文件初稿.docx",
            # "./file/成品软件类招标文件初稿1.docx",
            # "./file/服务类项目招标文件初稿1.docx",
            # "./file/服务类项目招标文件初稿2.docx",
            # "./file/物业类项目招标文件初稿.docx",
            # "./file/设备类招标文件初稿1.docx",
            # "./file/设备类招标文件初稿2.docx",
            # "./file/货物需求.docx"
        ]

        # 处理每个文件
        for docx_path in docx_path_files:
            try:
                if not os.path.exists(docx_path):
                    print(f"警告：文件不存在: {docx_path}")
                    continue
                    
                print(f"\n开始处理文件: {docx_path}")
                print("开始提取文档内容...")
                content_chunks = extract_docx_content(docx_path)
                print(f"文档分割为 {len(content_chunks)} 个块")
                
                # 检查每个内容块
                results = []
                total_chunks = len(content_chunks)
                for i, chunk in enumerate(content_chunks, 1):
                    print(f"正在处理第 {i}/{total_chunks} 个内容块...")
                    result = check_content(rule_processor, chunk)
                    if result:
                        results.append(result)
                
                # 生成报告文件名
                report_name = f"审查报告0520_{os.path.basename(docx_path).replace('.docx', '.txt')}"
                save_report(results, report_name)
                print(f"文件 {docx_path} 审查完成，共发现 {len(results)} 个问题。详细信息请查看 {report_name}")
                
            except Exception as e:
                print(f"处理文件 {docx_path} 时出错: {str(e)}")
                continue
        
        print("\n所有文件处理完成！")
        
    except Exception as e:
        print(f"程序执行出错: {str(e)}")

if __name__ == "__main__":
    main()
