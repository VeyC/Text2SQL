# 使用llm直接生成output

import copy
import json
import os
import re
import time
from typing import List
from openai import OpenAI
import sqlglot
import sqlite3
from typing_extensions import Annotated


def execute_sql(sql: Annotated[str, "SQL statements that need to be executed"]):
    print("=========== enter tool ===========")
    print(sql)
    conn = sqlite3.connect("/media/hnu/hnu2024/wangqin/python_work/OpenSearch-SQL/Bird/dev/dev_databases/california_schools/california_schools.sqlite") 
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        rows = cursor.fetchall()
        len_rows = len(rows)
        result = f"""The SQL statement:
{sql}
Execution successful.
The execution returned {len_rows} rows. 
The first eight rows: 
{rows[:8]}
""" 
        return "success", result
    except Exception as e:
        result = f"""The SQL statement:
{sql}
Execution failed.
The error message:
{e}
"""            
        return "failed", result


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
                    temperature=0.01,  # 使用较低的temperature获得更稳定的输出
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                print(f"调用LLM时发生错误 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    return f"Error: Failed to get response after {max_retries} attempts"
   
    def call_llm_with_tool(self, messages: list, max_retries: int = 3) -> str:
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
                    temperature=0.01,  # 使用较低的temperature获得更稳定的输出
                    tools=[
                        {
                        "type": "function",
                        "function": {
                            "name": "execute_sql",
                            "description": "执行 SQL 并返回结果",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "sql": {
                                        "type": "string",
                                        "description": "要执行的 SQL 语句"
                                    }
                                },
                                "required": ["sql"]
                            }
                        }
                    }
                    ],
                    tool_choice='required',
                )
                
                choice = response.choices[0].message

                if choice.tool_calls:  # 工具调用模式
                    print('==========================')
                    print(len(choice.tool_calls))
                    for tool_call in choice.tool_calls:
                        if tool_call.function.name == "execute_sql":
                            args = json.loads(tool_call.function.arguments)
                            sql_query = args.get("sql")
                            result = execute_sql(sql_query)  # 调用你的方法
                            print("执行结果:", result)
                            messages.append({
                                                "role": "function",
                                                "name": "execute_sql",
                                                "content": result[0]+'\n'+result[1]},
                                            )
                    respond = self.call_llm(messages)
                    return respond
                else:  # 普通对话模式
                    return choice.content
                
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


    def count_none_by_column(self, rows):
        """统计每列的None值数量"""
        if not rows:
            return {}
        
        # 获取列数
        num_columns = len(rows[0])
        
        # 初始化每列的None计数器
        column_none_counts = {col_idx: 0 for col_idx in range(num_columns)}
        
        # 统计每列的None值
        for row in rows:
            for col_idx, value in enumerate(row):
                if value is None:
                    column_none_counts[col_idx] += 1
        
        return column_none_counts

    
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
            
            # test_ids = [4, 16, 17, 23, 25, 26, 27, 28, 31, 33, 36, 37, 41, 43, 49, 50, 53, 54, 57, 58, 59, 63, 65, 72, 77, 80, 81, 84, 85, 86, 87, 88]
            # test_ids = [85]
            for i, item in enumerate(data):
                # if i not in test_ids:
                #     continue
                # if i+1 != 5:
                #     continue
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
                content_1 = f"""
Database Engine:
SQLite

Database Schema:
{item["db_desc"]}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{item["question"]}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
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
                            "content": content_1
                        }
                    ]
                llm_response = self.call_llm(messages)
                # 提取SQL语句
                pred_sqls = self.extract_sql_from_text(llm_response)

                # print('第一次调用#####################'*5)
                # print("输入：")
                # print(content_1)
                # print("输出：")
                # print(llm_response)
                
                # 为生成的sql定制knowledge
                knowledge_list = item['fd_text'].split('.\n')
                expression = sqlglot.parse_one(pred_sqls[-1], dialect='sqlite')
                columns = expression.find_all(sqlglot.exp.Column)
                columns_used = set(str(col) for col in columns)
                print("所有涉及的列：", columns_used)
                new_knowledge_list = []
                new_consistency_redundant_columns_list = []
                new_inconsistency_redundant_columns_list = []
                new_null_column_list = []
                for t_column in columns_used:
                    try:
                        _, t_column = t_column.replace('"', '`').split('.')
                    except:
                        t_column = t_column.replace('"', "`")
                    print(t_column)
                    new_knowledge_list.extend([knowledge for knowledge in knowledge_list if 'CDSCode' not in t_column and 'cds' not in t_column and ('.'+ t_column+ ' ' in knowledge or '.`' + t_column+'`' in knowledge)])
                    new_consistency_redundant_columns_list.extend([f"{A} and {B}\n" for A,B,C,D in item['consistency_redundant_columns'] if t_column in A+B])
                    new_inconsistency_redundant_columns_list.extend([f"{A} and {B}\n" for A,B, C,D in item['inconsistency_redundant_columns'] if t_column in A+B])
                    new_null_column_list.extend([col for col in item['null_column'] if t_column in col]) 

                new_knowledge_list = list(set(new_knowledge_list))
                new_consistency_redundant_columns_list = list(set(new_consistency_redundant_columns_list))
                new_inconsistency_redundant_columns_list = list(set(new_inconsistency_redundant_columns_list))
                new_null_column_list = list(set(new_null_column_list))

                # 第二次调用LLM，加knowledge确认
                content_2 = f"""I give you the following additional knowledge and constrains. Please reconsider the SQL you generated before. You must strictly check and ensure that every constraint is satisfied.'

Knowledge:
In addition, **there are some redundant columns here, but their stored data is consistent. You can use one of them freely.**
""" + ' '.join(new_consistency_redundant_columns_list) + """

**There are also some redundant columns, but the data they store is inconsistent. When querying involving these columns, you need to carefully consider which column to use.**
""" + ' '.join(new_inconsistency_redundant_columns_list) + """

**The following columns contain NULL values. When these columns are involved in SQL queries, you should carefully consider the user's intent and determine whether explicit NOT NULL checks are necessary.
""" + '\n'.join(new_null_column_list) + """

Constraints:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- You should generate an executable SQL statement. Multiple queries is not allowed.
- You may perform a **NOT NULL** check on the relevant columns based on the user's question, especially when these columns contain null values.
- You must further understand user intent based on these cardinality relationships, especially when there is an N:1 ratio between two attributes, it can affect how GROUP BY and ORDER BY are used.
- When retrieving the top N rows in a query with JOIN, make sure to consider the order in which ORDER BY and JOIN are applied, because applying ORDER BY too early or on only one table can lead to incorrect results.
"""

# 去掉这部分试试
# The following are the cardinality relationships between attributes. For example, the cardinality relationships between student ID and age is N:1, as there may be multiple students with the same age, but there is no student ID with multiple ages.
# """ + '.\n'.join(new_knowledge_list) + """

                messages.append({"role": "assistant", "content": pred_sqls[-1]})
                # messages.append({"role": "assistant", "content": llm_response})
                messages.append({"role": "user", "content": content_2})
                second_llm_response = self.call_llm(messages)
                # 提取SQL语句
                second_pred_sqls = self.extract_sql_from_text(second_llm_response)
                final_pred_sqls = copy.deepcopy(second_pred_sqls)
                # third_llm_response = copy.deepcopy(second_llm_response)
                # content_3 = copy.deepcopy(content_2)
                # print('第二次调用#####################'*5)
                # print("输入：")
                # print(content_2)
                # print("输出：")
                # print(second_llm_response)

                # assert 1==0

                #=============================== 先不加吧，这个运行要好久 =======================#
#                 # 添加后执行，第三次调用LLM
#                 conn = sqlite3.connect("/media/hnu/hnu2024/wangqin/python_work/OpenSearch-SQL/Bird/dev/dev_databases/california_schools/california_schools.sqlite") 
#                 cursor = conn.cursor()
#                 try:
#                     cursor.execute(second_pred_sqls[-1])
#                     rows = cursor.fetchall()
#                     len_rows = len(rows)
#                     print('execute sucess =========================')
#                     print(rows[:5])

#                     if len_rows == 0:
#                         content_3 = f"""Based on your previously generated SQL:
# {second_pred_sqls[-1]}
# the execution returned {len_rows} rows. This is incorrect.

# You should revise the SQL based on the user's question and the execution results, such as removing some NOT NULL check which is not neccessary.
# """
#                         messages.append({"role": "assistant", "content": second_pred_sqls[-1]})
#                         # messages.append({"role": "assistant", "content": llm_response})
#                         messages.append({"role": "user", "content": content_3})
#                         third_llm_response = self.call_llm(messages)
#                         # 提取SQL语句
#                         final_pred_sqls = self.extract_sql_from_text(third_llm_response)

#                         print('第三次调用#####################'*5)
#                         print("输入：")
#                         print(content_3)
#                         print("输出：")
#                         print(third_llm_response)


#                     elif any(None in row for row in rows): # 存在none值
#                         column_none_counts = self.count_none_by_column(rows) #dict
#                         content_3 = f"""Based on your previously generated SQL:
# {second_pred_sqls[-1]}
# the execution returned {len_rows} rows, with the first five rows being: 
# {rows[:5]}
# For the following columns (indexed starting from 1), the number of None values are respectively: 
# """ + '.\n'.join([f'column {col+1} have {count} None value' for col, count in column_none_counts.items()]) + """

# You can revise the SQL based on the user's question and the execution results, adding NOT NULL checks to the relevant columns when **necessary** to avoid None values.
# """
#                         messages.append({"role": "assistant", "content": second_pred_sqls[-1]})
#                         # messages.append({"role": "assistant", "content": llm_response})
#                         messages.append({"role": "user", "content": content_3})
#                         third_llm_response = self.call_llm(messages)
#                         # 提取SQL语句
#                         final_pred_sqls = self.extract_sql_from_text(third_llm_response)

#                         print('第三次调用#####################'*5)
#                         print("输入：")
#                         print(content_3)
#                         print("输出：")
#                         print(third_llm_response)

#                 except Exception as e:
#                     print('execute error =======================')
#                     print(e)
#                     content_3 = f"""Based on your previously generated SQL:
# {second_pred_sqls[-1]} 
# the following error occurred upon execution: 
# {e}

# You should revise the SQL according to the user's question and the execution results to ensure it runs correctly."""
#                     messages.append({"role": "assistant", "content": second_pred_sqls[-1]})
#                     # messages.append({"role": "assistant", "content": llm_response})
#                     messages.append({"role": "user", "content": content_3})
#                     third_llm_response = self.call_llm(messages)
#                     # 提取SQL语句
#                     final_pred_sqls = self.extract_sql_from_text(third_llm_response)
#                     print('第三次调用#####################'*5)
#                     print("输入：")
#                     print(content_3)
#                     print("输出：")
#                     print(third_llm_response)


                # 构建输出记录
                output_item = {
                    "input_seq": item.get("input_seq", ""),
                    "db_desc": item.get("db_desc", ""),
                    "question": item.get("question", ""),
                    "output_seq": item.get("output_seq", ""),
                    "pred_sqls_round1":pred_sqls,
                    "pred_sqls_round2": second_pred_sqls,
                    "final_pred_sqls": final_pred_sqls,
                    "responses": [llm_response, second_llm_response],
                    "first_call_input": content_1,
                    "second_call_input": content_2
                }
                
                processed_data.append(output_item)
                
                # 添加SQL到总列表（取第一个SQL，如果存在的话）
                if final_pred_sqls:
                    all_pred_sqls.append(final_pred_sqls[-1])
                else:
                    all_pred_sqls.append("")  # 如果没有提取到SQL，添加空字符串
                
                print(f"  - 提取到 {len(final_pred_sqls)} 个SQL语句")
                print(final_pred_sqls[-1])

                
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


    def parse_action_from_response(self, response: str):
        """
        提取最后一个ActionInput中的函数调用
        格式：execute_sql(sql="...")
        """
        
        # 找到所有的ActionInput块
        action_input_pattern = r'ActionInput:\s*(.+?)(?=\nObservation:|\nThink:|\nAction:|\nFinal Answer:|$)'
        matches = re.findall(action_input_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if not matches:
            return None, None
        
        # 取最后一个ActionInput
        last_action_input = matches[-1].strip()
        
        # 提取函数调用：execute_sql(sql="...")
        func_pattern = r'execute_sql\s*\(\s*sql\s*=\s*["\'](.+?)["\']'
        func_match = re.search(func_pattern, last_action_input, re.DOTALL)
        
        if func_match:
            sql_query = func_match.group(1).strip()
            return "execute_sql", sql_query
        
        return None, None


    def call_llm_with_tool_react(self, messages: list, max_iterations: int = 4) -> str:
        """
        使用ReAct格式的工具调用
        """
        current_messages = copy.deepcopy(messages)
        iteration = 0
        
        while iteration < max_iterations:
            # 注意：这里不使用tools参数，让模型纯文本输出
            response = self.call_llm(current_messages)
            
            print(f"=== 迭代 {iteration + 1} ===")
            print("模型输出：")
            print(response)
            
            # 检查是否包含Final Answer
            if "Final Answer:" in response:
                return response
                
            # 解析Action和ActionInput
            action, action_input = self.parse_action_from_response(response)
            
            if action and action_input:
                if action.strip() == "execute_sql":
                    # 执行SQL
                    result = execute_sql(action_input.strip())
                    observation = result
                    
                    print(f"执行SQL: {action_input}")
                    print(f"观察结果: {observation}")
                    
                    # 将观察结果添加到消息中
                    updated_response = response + f"\nObservation: {observation}"
                    current_messages.append({"role": "assistant", "content": updated_response})
                    
                    # 继续对话，让模型基于观察结果继续推理
                    continue_prompt = "Based on the observation above, continue your reasoning. What should you do next?"
                    current_messages.append({"role": "user", "content": continue_prompt})
                else:
                    break
            else:
                break
                    
            iteration += 1
        
        # 如果达到最大迭代次数，进行最后一次调用要求给出最终答案
        final_prompt = "Please provide your Final Answer now based on all the observations above."
        current_messages.append({"role": "user", "content": final_prompt})
        final_response = self.call_llm(current_messages)
        
        return final_response



    def process_json_file_two_stage_agent(self, input_file: str, output_file: str, sql_file: str):
        """
        处理JSON文件，调用LLM并提取SQL, 这里的llm是可以调用工具的。
        
        Args:
            input_file: 输入的JSON文件路径
            output_file: 输出的完整结果文件路径
            sql_file: 输出的SQL列表文件路径
        """
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"读取到 {len(data)} 条记录")
        
        # 处理结果存储
        processed_data = []
        all_pred_sqls = []
        
        test_ids = [85]
        for i, item in enumerate(data):
            if i not in test_ids:
                continue
            # if i+1 != 5:
            #     continue
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
            content_1 = f"""
Database Engine:
SQLite

Database Schema:
{item["db_desc"]}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{item["question"]}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.
- When it comes to division, convert the denominator to a float.

Output Format:
{instruct_info}
    """.strip()
                
            # 第一次调用LLM，不加knowledge，主要是用来筛选column的
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
                            "content": content_1
                        }
                    ]
            llm_response = self.call_llm(messages)
            # 提取SQL语句
            pred_sqls = self.extract_sql_from_text(llm_response)

            print('第一次调用#####################'*5)
            print("输入：")
            print(content_1)
            print("输出：")
            print(llm_response)
            
            # 为生成的sql定制knowledge
            knowledge_list = item['fd_text'].split('.\n')
            expression = sqlglot.parse_one(pred_sqls[-1], dialect='sqlite')
            columns = expression.find_all(sqlglot.exp.Column)
            columns_used = set(str(col) for col in columns)
            print("所有涉及的列：", columns_used)
            new_knowledge_list = []
            new_consistency_redundant_columns_list = []
            new_inconsistency_redundant_columns_list = []
            new_null_column_list = []
            for t_column in columns_used:
                try:
                    _, t_column = t_column.replace('"', '`').split('.')
                except:
                    t_column = t_column.replace('"', "`")
                print(t_column)
                new_knowledge_list.extend([knowledge for knowledge in knowledge_list if 'CDSCode' not in t_column and 'cds' not in t_column and ('.'+ t_column+ ' ' in knowledge or '.`' + t_column+'`' in knowledge)])
                new_consistency_redundant_columns_list.extend([f"{A} and {B}\n" for A,B,C,D in item['consistency_redundant_columns'] if t_column in A+B])
                new_inconsistency_redundant_columns_list.extend([f"{A} and {B}\n" for A,B, C,D in item['inconsistency_redundant_columns'] if t_column in A+B])
                new_null_column_list.extend([col for col in item['null_column'] if t_column in col]) 

            new_knowledge_list = list(set(new_knowledge_list))
            new_consistency_redundant_columns_list = list(set(new_consistency_redundant_columns_list))
            new_inconsistency_redundant_columns_list = list(set(new_inconsistency_redundant_columns_list))
            new_null_column_list = list(set(new_null_column_list))

            # 第二次调用LLM，加knowledge确认
            content_2 = f"""I give you the following additional knowledge and constrains. Please reconsider the SQL you generated before. You must strictly check and ensure that every constraint is satisfied.'

# Knowledge:
The following are the cardinality relationships between attributes. For example, the cardinality relationships between student ID and age is N:1, as there may be multiple students with the same age, but there is no student ID with multiple ages.
""" + '.\n'.join(new_knowledge_list) + """

In addition, **there are some redundant columns here, but their stored data is consistent. You can use one of them freely.**
""" + ' '.join(new_consistency_redundant_columns_list) + """

**There are also some redundant columns, but the data they store is inconsistent. When querying involving these columns, you need to carefully consider which column to use.**
""" + ' '.join(new_inconsistency_redundant_columns_list) + """

**The following columns contain NULL values. When these columns are involved in SQL queries, you should carefully consider the user's intent and determine whether explicit NOT NULL checks are necessary.
""" + '\n'.join(new_null_column_list) + """

# Constraints:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- You should generate an executable SQL statement. Multiple queries is not allowed.
- You may perform a **NOT NULL** check on the relevant columns based on the user's question, especially when these columns contain null values.
- You must further understand user intent based on these cardinality relationships, especially when there is an N:1 ratio between two attributes, it can affect how GROUP BY and ORDER BY are used.
- When retrieving the top N rows in a query with JOIN, make sure to consider the order in which ORDER BY and JOIN are applied, because applying ORDER BY too early or on only one table can lead to incorrect results.

# Notice
- A well-crafted SQL query is not created in one go; it often requires continuous adjustments to meet the user's needs. Since you have no knowledge of the actual stored data, you can first generate some original SQL queries (not the final SQL given to the user) and execute them via a tool to examine the data you are interested in. Especially when you have difficulty or ambiguity in understanding user needs, this data will provide you with key information, which is crucial for generating the final SQL query that meets user needs.
- You can only use the actions provided in the **Action space** to solve the task.
- IMPORTANT: After writing ActionInput, STOP generating. Wait for the system to provide the Observation. Do NOT generate the Observation yourself.

# Action space: 
## execute_sql Action
* Signature: execute_sql(sql="sql statement") 
* Description: This action string will execute in the sqlite engine. 
* Example: execute_sql(sql="select ID from student;")


# Output:
Answer the following questions to the best of your ability.
Use the following format:
Requirement: The requirement you need to follow
Think: You should always think about what to do
Action: The action to take
ActionInput: The input for the action
Observation: The result of the action. Wait for the Observation before proceeding. 
…(This process can be repeated no more than 4 times)
Think: I now know the final answer
Final Answer: The final answer to the original input question
```sql
Correct SQL query here
```

Get started!
Requirement: {Input}
"""
            messages.append({"role": "assistant", "content": pred_sqls[-1]})
            messages.append({"role": "user", "content": content_2})


            second_llm_response = self.call_llm_with_tool_react(messages)
            
            # third_llm_response = copy.deepcopy(second_llm_response)
            # content_3 = copy.deepcopy(content_2)
            print('第二次调用#####################'*5)
            print("输入：")
            print(content_2)
            print("输出：")
            print(second_llm_response)
            assert 1==0

            # 提取SQL语句
            second_pred_sqls = self.extract_sql_from_text(second_llm_response)
            final_pred_sqls = copy.deepcopy(second_pred_sqls)

            

              
            # 构建输出记录
            output_item = {
                "input_seq": item.get("input_seq", ""),
                "db_desc": item.get("db_desc", ""),
                "question": item.get("question", ""),
                "output_seq": item.get("output_seq", ""),
                "pred_sqls_round1":pred_sqls,
                "pred_sqls_round2": second_pred_sqls,
                "final_pred_sqls": final_pred_sqls,
                "responses": [llm_response, second_llm_response],
                "first_call_input": content_1,
                "second_call_input": content_2
            }
            
            processed_data.append(output_item)
            
            # 添加SQL到总列表（取第一个SQL，如果存在的话）
            if final_pred_sqls:
                all_pred_sqls.append(final_pred_sqls[-1])
            else:
                all_pred_sqls.append("")  # 如果没有提取到SQL，添加空字符串
            
            print(f"  - 提取到 {len(final_pred_sqls)} 个SQL语句")
            print(final_pred_sqls[-1])

            
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



def main():
    """
    主函数 - 使用示例
    """
    # 配置参数
    API_KEY = os.getenv('OPENAI_API_KEY') # 替换为您的API密钥
    BASE_URL = "https://www.dmxapi.com/v1/"  # 如果使用OpenAI官方API，保持为None；如果使用其他兼容的API，设置相应的URL
    MODEL = "gpt-4o"  # 可以改为其他模型，如 "gpt-4o" gemini-2.5-pro

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
    extractor.process_json_file_two_stage_agent(INPUT_FILE, OUTPUT_FILE, SQL_FILE) # 一阶段直接生成sql

if __name__ == "__main__":
    main()

    #TODO 无语了，先测第五个实例。

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