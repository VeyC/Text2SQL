# 使用llm直接生成output

import json
import os
import re
import time
from typing import List
from openai import OpenAI
import sqlglot

class LLMSQLExtractor:
    def __init__(self, api_key: str, base_url: str = None, model: str = "gpt-3.5-turbo"):
        """
        初始化LLM客户端
        
        Args:
            api_key: OpenAI API密钥
            base_url: 可选的基础URL，用于其他兼容OpenAI的API
            model: 使用的模型名称
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        
    def extract_sql_from_text(self, text: str) -> List[str]:
        """
        从文本中提取SQL语句
        
        Args:
            text: 包含SQL的文本
            
        Returns:
            提取出的SQL语句列表
        """
        # 匹配 ```sql 和 ``` 之间的内容
        sql_pattern = r'```sql\s*(.*?)\s*```'
        sql_matches = re.findall(sql_pattern, text, re.DOTALL | re.IGNORECASE)
        
        # 清理提取的SQL语句
        cleaned_sqls = []
        for sql in sql_matches:
            # 去除多余的空白字符
            cleaned_sql = sql.strip()
            if cleaned_sql:
                cleaned_sqls.append(cleaned_sql)
        
        return cleaned_sqls
    
    def call_llm(self, messages: list, max_retries: int = 3) -> str:
        """
        调用大模型生成回复
        
        Args:
            input_text: 输入文本
            max_retries: 最大重试次数
            
        Returns:
            模型的回复
        """
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1  # 使用较低的temperature获得更稳定的输出
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                print(f"调用LLM时发生错误 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    return f"Error: Failed to get response after {max_retries} attempts"
    
    def process_json_file(self, input_file: str, output_file: str, sql_file: str):
        """
        处理JSON文件，调用LLM并提取SQL
        
        Args:
            input_file: 输入的JSON文件路径
            output_file: 输出的完整结果文件路径
            sql_file: 输出的SQL列表文件路径
        """
        try:
            # 读取输入文件
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"读取到 {len(data)} 条记录")
            
            # 处理结果存储
            processed_data = []
            all_pred_sqls = []
            
            for i, item in enumerate(data):
                print(f"处理第 {i + 1}/{len(data)} 条记录...")
                
                instruct_info = """
Please provide a detailed chain-of-thought reasoning process and include your thought process within `<think>` tags. Your final answer should be enclosed within `<answer>` tags.

Ensure that your SQL query follows the correct syntax and is formatted as follows:

```sql
-- Your SQL query here
```

Example format:
<think> Step-by-step reasoning, including self-reflection and corrections if necessary. [Limited by 4K tokens] </think>
<answer> Summary of the thought process leading to the final SQL query. [Limited by 1K tokens]

```sql
Correct SQL query here
```
</answer>""".strip()
                content = f"""
Database Engine:
SQLite

Database Schema:
{item["db_desc"]}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{item["question"]}

Knowledge:
{item['knowledge']}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- Try not to use 'where value=select max (colunm)' as much as possible, instead use 'order by'`
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.
- When it comes to division, convert the denominator to a float.

Output Format:
{instruct_info}
    """.strip()
                messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are a data science expert. Below, you are provided with a database schema and a natural"
                                " language question. Your task is to understand the schema and generate a valid SQL query to"
                                " answer the question."
                            ),
                        },
                        {
                            "role": "user", 
                            "content": content
                        }
                    ]
                # 调用LLM
                llm_response = self.call_llm(messages)
                
                # 提取SQL语句
                pred_sqls = self.extract_sql_from_text(llm_response)
                
                # 构建输出记录
                output_item = {
                    "input_seq": item.get("input_seq", ""),
                    "output_seq": item.get("output_seq", ""),
                    "db_desc": item.get("db_desc", ""),
                    "question": item.get("question", ""),
                    "responses": [llm_response],
                    "pred_sqls": pred_sqls
                }
                
                processed_data.append(output_item)
                
                # 添加SQL到总列表（取第一个SQL，如果存在的话）
                if pred_sqls:
                    all_pred_sqls.append(pred_sqls[0])
                else:
                    all_pred_sqls.append("")  # 如果没有提取到SQL，添加空字符串
                
                print(f"  - 提取到 {len(pred_sqls)} 个SQL语句")
                print(pred_sqls[0])
                
                # 添加延迟避免API限流
                time.sleep(0.5)
            
            # 保存完整结果
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
            # 保存SQL列表
            with open(sql_file, 'w', encoding='utf-8') as f:
                json.dump(all_pred_sqls, f, ensure_ascii=False, indent=2)
            
            print(f"\n处理完成!")
            print(f"完整结果已保存到: {output_file}")
            print(f"SQL列表已保存到: {sql_file}")
            print(f"总共处理了 {len(processed_data)} 条记录")
            print(f"提取了 {len([sql for sql in all_pred_sqls if sql])} 个有效SQL语句")
            
        except FileNotFoundError:
            print(f"错误: 找不到输入文件 {input_file}")
        except json.JSONDecodeError:
            print(f"错误: {input_file} 不是有效的JSON文件")
        except Exception as e:
            print(f"处理过程中发生错误: {str(e)}")

    
    def process_json_file_two_stage(self, input_file: str, output_file: str, sql_file: str):
        """
        处理JSON文件，调用LLM并提取SQL
        
        Args:
            input_file: 输入的JSON文件路径
            output_file: 输出的完整结果文件路径
            sql_file: 输出的SQL列表文件路径
        """
        try:
            # 读取输入文件
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"读取到 {len(data)} 条记录")
            
            # 处理结果存储
            processed_data = []
            all_pred_sqls = []
            
            for i, item in enumerate(data):
                print(f"处理第 {i + 1}/{len(data)} 条记录...")
                
                instruct_info = """
Please provide a detailed chain-of-thought reasoning process and include your thought process within `<think>` tags. Your final answer should be enclosed within `<answer>` tags.

Ensure that your SQL query follows the correct syntax and is formatted as follows:

```sql
-- Your SQL query here
```

Example format:
<think> Step-by-step reasoning, including self-reflection and corrections if necessary. [Limited by 4K tokens] </think>
<answer> Summary of the thought process leading to the final SQL query. [Limited by 1K tokens]

```sql
Correct SQL query here
```
</answer>""".strip()
                content = f"""
Database Engine:
SQLite

Database Schema:
{item["db_desc"]}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{item["question"]}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- Try not to use 'where value=select max (colunm)' as much as possible, instead use 'order by'`
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.
- When it comes to division, convert the denominator to a float.

Output Format:
{instruct_info}
    """.strip()
                
                # 第一次调用LLM，不加knowledge
                messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are a data science expert. Below, you are provided with a database schema and a natural"
                                " language question. Your task is to understand the schema and generate a valid SQL query to"
                                " answer the question."
                            ),
                        },
                        {
                            "role": "user", 
                            "content": content
                        }
                    ]
                llm_response = self.call_llm(messages)
                # 提取SQL语句
                pred_sqls = self.extract_sql_from_text(llm_response)
                
                # 为生成的sql定制knowledge
                knowledge_list = item['fd_text'].split('.\n')
                expression = sqlglot.parse_one(pred_sqls[-1], dialect='sqlite')
                columns = expression.find_all(sqlglot.exp.Column)
                columns_used = set(str(col) for col in columns)
                print("所有涉及的列：", columns_used)
                new_knowledge_list = []
                new_consistency_redundant_columns_list = []
                new_inconsistency_redundant_columns_list = []
                for t_colunm in columns_used:
                    t_colunm = t_colunm.replace('"', "`")
                    print(t_colunm)
                    new_knowledge_list.extend([knowledge for knowledge in knowledge_list if t_colunm in knowledge])
                    new_consistency_redundant_columns_list.extend([f"{A} and {B}\n" for A,B,C,D in item['consistency_redundant_columns'] if t_colunm in A+B])
                    new_inconsistency_redundant_columns_list.extend([f"{A} and {B}\n" for A,B, C,D in item['inconsistency_redundant_columns'] if t_colunm in A+B])

                new_knowledge_list = list(set(new_knowledge_list))
                new_consistency_redundant_columns_list = list(set(new_consistency_redundant_columns_list))
                new_inconsistency_redundant_columns_list = list(set(new_inconsistency_redundant_columns_list))

                # 第二次调用LLM，加knowledge确认
                content = f"""I give you the following additional knowledge and instruction rules. Please reconsider the SQL you generated before.

Knowledge:
The following are the cardinality relationships between attributes. A many-to-one (N:1) relationship exists between attribute A and attribute B ([A, B]) if A functionally determines B (i.e., A → B), but B does not functionally determine A. In this case, each value of A can be associated with multiple values of B, but each value of B corresponds to exactly one value of A. A one-to-one (1:1) relationship exists between attribute A and attribute B if both A → B and B → A hold. This means each value of A corresponds to exactly one value of B, and vice versa. Any attribute pairs not mentioned are assumed to have a many-to-many (N:N) relationship. For example, the ratio between student ID and age is N:1, as there may be multiple students with the same age.
""" + '.\n'.join(new_knowledge_list) + """

In addition, **there are some redundant columns here, but their stored data is consistent. You can use one of them freely.**
""" + '\n'.join(new_consistency_redundant_columns_list) + """

**There are also some redundant columns, but the data they store is inconsistent. When querying involving these columns, you need to carefully consider which column to use.**
""" + '\n'.join(new_inconsistency_redundant_columns_list) + """

Instructions:
- You should generate an executable SQL statement. Multiple queries is not allowed.
- You should check if all columns are in the correct table. Incorrect positioning will directly cause SQL to fail to run.
- You should perform a NOT NULL check on the field when you want to select it or use it in an ORDER BY clause or sort it in ASC or DESC order.
"""
                messages.append({"role": "assistant", "content": pred_sqls[-1]})
                messages.append({"role": "user", "content": content})
                second_llm_response = self.call_llm(messages)
                # 提取SQL语句
                second_pred_sqls = self.extract_sql_from_text(second_llm_response)

                # 构建输出记录
                output_item = {
                    "input_seq": item.get("input_seq", ""),
                    "output_seq": item.get("output_seq", ""),
                    "db_desc": item.get("db_desc", ""),
                    "question": item.get("question", ""),
                    "responses": [llm_response, second_llm_response],
                    "pred_sqls": second_pred_sqls
                }
                
                processed_data.append(output_item)
                
                # 添加SQL到总列表（取第一个SQL，如果存在的话）
                if second_pred_sqls:
                    all_pred_sqls.append(second_pred_sqls[0])
                else:
                    all_pred_sqls.append("")  # 如果没有提取到SQL，添加空字符串
                
                print(f"  - 提取到 {len(second_pred_sqls)} 个SQL语句")
                print(second_pred_sqls[0])
                
                # 添加延迟避免API限流
                time.sleep(0.5)
            
            # 保存完整结果
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
            # 保存SQL列表
            with open(sql_file, 'w', encoding='utf-8') as f:
                json.dump(all_pred_sqls, f, ensure_ascii=False, indent=2)
            
            print(f"\n处理完成!")
            print(f"完整结果已保存到: {output_file}")
            print(f"SQL列表已保存到: {sql_file}")
            print(f"总共处理了 {len(processed_data)} 条记录")
            print(f"提取了 {len([sql for sql in all_pred_sqls if sql])} 个有效SQL语句")
            
        except FileNotFoundError:
            print(f"错误: 找不到输入文件 {input_file}")
        except json.JSONDecodeError:
            print(f"错误: {input_file} 不是有效的JSON文件")
        except Exception as e:
            print(f"处理过程中发生错误: {str(e)}")



def main():
    """
    主函数 - 使用示例
    """
    # 配置参数
    API_KEY = os.getenv('OPENAI_API_KEY') # 替换为您的API密钥
    BASE_URL = "https://www.dmxapi.com/v1/"  # 如果使用OpenAI官方API，保持为None；如果使用其他兼容的API，设置相应的URL
    MODEL = "gpt-4o"  # 可以改为其他模型，如 "gpt-4"

    # 文件路径
    INPUT_FILE = "../output/bird/dev_bird.json"
    # INPUT_FILE = "/media/hnu/hnu2024/wangqin/python_work/ArcticTraining/projects/arctic_text2sql_r1/Bird/dev_bird_colifornia_school.json"
    OUTPUT_FILE = "../output/bird/llm_predict_base_two_stage.json"
    SQL_FILE = "../output/bird/llm_predict_sql_two_stage.json"
    
    # 创建提取器实例
    extractor = LLMSQLExtractor(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL
    )
    
    # 处理文件
    extractor.process_json_file_two_stage(INPUT_FILE, OUTPUT_FILE, SQL_FILE) # 一阶段直接生成sql

if __name__ == "__main__":
    main()

# 如果需要自定义使用，可以这样调用：
"""
from llm_sql_extractor import LLMSQLExtractor

# 初始化
extractor = LLMSQLExtractor(
    api_key="your-api-key",
    model="gpt-4"
)

# 处理文件
extractor.process_json_file(
    input_file="dev.json",
    output_file="results.json", 
    sql_file="pred_sqls.json"
)
"""