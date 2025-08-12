# 只测试一个表, 先只生成sql，后面再用别人的代码评估执行准确率
import ijson
import argparse
import random
import os
from tqdm import tqdm
from db_class import Database
from openai import OpenAI
import json
import re


def load_json_file(file, database_name):
    dataset = []
    with open(file, "r", encoding="utf-8") as f:
        objects = ijson.items(f, "item")
        for obj in tqdm(objects):
            if obj["db_id"] == database_name:
                dataset.append(obj)
    return dataset




if __name__ == "__main__":
    # 加载参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_file", type=str, default="../datasets/bird/dev/dev.json", help="输入包含问题的数据文件")
    parser.add_argument("--output_data_file", type=str, default="../output/bird/dev_bird.json", help="输出数据文件，包含input prompt等，它将作为model的输入")
    parser.add_argument("--db_path", type=str, default="../datasets/bird/dev/dev_databases/", help="数据库路径")
    parser.add_argument("--tables", type=str, default="../datasets/bird/dev/dev_tables.json", help="表定义文件")
    parser.add_argument("--database", type=str, default="california_schools", help="数据库名称, 因为我们先测一个数据库")
    parser.add_argument("--source", type=str, default="bird")
    parser.add_argument("--model_name", type=str, default="gemini-2.5-pro")
    parser.add_argument("--mode", type=str, default="dev")
    parser.add_argument("--value_limit_num", type=int, default=2, help="每个表的值限制数量")
    parser.add_argument("--schema_description_path", type=str, default="../datasets/bird/dev/dev_bird.json", help="提供初始DDL描述")
    # parser.add_argument("--db_content_index_path", type=str, default="./data/bird/dev_20240627/db_contents_index")  试试不加索引，看效果
    # 加载数据
    opt = parser.parse_args()
    print(opt)

    random.seed(42)
    assert opt.mode in ["train", "dev", "test"]
    dataset = load_json_file(opt.input_data_file, opt.database)
    print(f"加载了{len(dataset)}条数据")

    # 处理数据库
    db_path = os.path.join(opt.db_path, opt.database, opt.database+'.sqlite')
    db_model = Database(db_path)
    tables = db_model.list_tables()
    
    database_ratio_maps = {}
    # 遍历所有表
    for table in tables:
        table_schema_dict = db_model.get_table_attrs(table)
        # 1.1 先对数据库中的每个表分析函数依赖关系
        try:
            # fd_list shape [[[Left],'Right'], [[Left], 'Right'], ...]
            fd_list = db_model.analyze_specific_table(table, max_lhs_size=1)  # 这个可以根据primary key的个数来定
            print(f"\n表 {table} 的分析完成\n" + "="*50)
        except Exception as e:
            fd_list = []
            print(f"分析表 {table} 时出错: {e}")
        # 1.2 这里的函数依赖太多了，需要过滤一些，首先是column中只有一个值的，这样的column谁都能确定，它一直被位于右值。
        table_distributions = db_model.analyze_column_distribution(table)
        filted_fd_list = []
        for fd in fd_list:
            left_attr = fd[0][0]  # 后面是[0]，因为之前设置的依赖size是1
            right_attr = fd[1]
            # 左值是pk的就不要了吧，毕竟这个函数依赖关系确定了
            if table_schema_dict[left_attr]['pk'] == False and table_distributions[right_attr]['unique_count'] > 1: 
                filted_fd_list.append([left_attr, right_attr])  # 这里去掉left的[]了
        
        # 2. 处理属性之间的比例关系。
        fd_set = set(tuple(fd) for fd in filted_fd_list)
        visited = set()
        ratio_map = {"1:1":[], "N:1":[]}

        for fd in filted_fd_list:
            left, right = fd
            if (left, right) in visited or (right, left) in visited:
                continue 

            reverse_fd = (right, left)
            if reverse_fd in fd_set:
                ratio_map["1:1"].append([left, right])
                visited.add((left, right))
                visited.add((right, left))
            else:
                ratio_map["N:1"].append([left, right])
                visited.add((left, right))

        database_ratio_maps[table] = ratio_map
    
    print(f"数据库属性比例关系：\n {database_ratio_maps}")
   
    DDL = db_model.get_database_ddls()

    # TODO，用语言描述是不是会好点
    prompt = f'''Task Overview:
Below, you are provided with a database schema and the proportional relationship between attributes in json format. Please describe the proportional relationship in natural language format. For example, "the relationship from 'satscores.sname' to 'satcores.rttype' is N:1, indicating that ... ." Each relationship requires line breaks, and there is no need to output additional analysis content beyond that.

Database Schema:
{DDL}

Proportional Relationship between Attributes:
{database_ratio_maps}
'''
    # client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url="https://www.dmxapi.com/v1/")
    # response = client.chat.completions.create(
    #         model=opt.model_name,
    #         # model = "deepseek-reasoner",
    #         messages=[
    #             {"role": "system", "content": "You are a data science expert."},
    #             {"role": "user", "content": prompt}
    #         ],
    #         stream=False
    #     )
    # database_ratio_text = response.choices[0].message.content
    # print(f"LLM 输入：\n{prompt}")
    # print(f"LLM 输出text：\n{database_ratio_text}")
    # assert 1==0
    database_ratio_text = """The relationship from frpm.County Name to frpm.County Code is 1:1, indicating that each county name corresponds to exactly one county code and vice versa.
The relationship from frpm.District Code to frpm.County Code is N:1, indicating that multiple district codes may be associated with the same county code.
The relationship from frpm.District Code to frpm.County Name is N:1, indicating that multiple district codes may share the same county name.
The relationship from frpm.District Code to frpm.District Name is N:1, indicating that multiple district codes may share the same district name.
The relationship from frpm.District Code to frpm.District Type is N:1, indicating that many districts may have the same type.
The relationship from frpm.School Code to frpm.School Name is N:1, indicating that multiple school codes may refer to the same school name.
The relationship from frpm.School Code to frpm.School Type is N:1, indicating that multiple school codes may share the same school type.
The relationship from frpm.School Code to frpm.Educational Option Type is N:1, indicating that many schools may fall under the same educational option type.
The relationship from frpm.School Code to frpm.NSLP Provision Status is N:1, indicating that multiple school codes may have the same NSLP status.
The relationship from frpm.School Code to frpm.Charter School (Y/N) is N:1, indicating that multiple school codes may share the same charter school status.
The relationship from frpm.School Code to frpm.Charter School Number is N:1, indicating that many schools may share the same charter number.
The relationship from frpm.School Code to frpm.Charter Funding Type is N:1, indicating that multiple schools may have the same charter funding type.
The relationship from frpm.School Code to frpm.IRC is N:1, indicating that several schools may share the same IRC value.
The relationship from frpm.District Name to frpm.District Type is N:1, indicating that multiple districts with the same name may share the same type.
The relationship from frpm.Charter School Number to frpm.Charter Funding Type is N:1, indicating that the same charter number may correspond to a single funding type.
The relationship from satscores.sname to satscores.rtype is N:1, indicating that multiple schools with the same name may belong to the same reporting type.
The relationship from schools.DOCType to schools.DOC is 1:1, indicating that each DOC type corresponds uniquely to a DOC code and vice versa.
The relationship from schools.SOCType to schools.SOC is 1:1, indicating that each SOC type corresponds uniquely to a SOC code and vice versa.
The relationship from schools.EdOpsName to schools.EdOpsCode is 1:1, indicating that each educational option name corresponds to a unique code and vice versa.
The relationship from schools.EILName to schools.EILCode is 1:1, indicating that each EIL name corresponds to a unique code and vice versa.
The relationship from schools.AdmEmail3 to schools.AdmLName3 is 1:1, indicating that each admin email corresponds to a unique last name and vice versa.
The relationship from schools.StreetAbr to schools.Street is N:1, indicating that multiple street abbreviations may refer to the same full street name.
The relationship from schools.MailStrAbr to schools.MailStreet is N:1, indicating that multiple abbreviated mailing street names may refer to the same full name.
The relationship from schools.AdmLName3 to schools.AdmFName3 is N:1, indicating that multiple admin last names may be associated with the same first name.
The relationship from schools.AdmEmail3 to schools.AdmFName3 is N:1, indicating that multiple emails may be associated with the same admin first name.
"""

    # 3. 处理冗余情况，和数据不一致情况，数据库设计规范的问题，这个很难从数值上确定是不是同一列，可能需要大模型，当然数值上也可以缩小范围
    
    prompt = f'''Task Overview:
Below, you are provided with a database schema. Your task is to understand the schema and identify redundant columns.

Database Schema:
{DDL}

Output Format:
In your answer, please enclose the generated SQL query in a code block:
```json
\[["table1.column1", "table2.column2", "table1.key1", "table2.key2"], ...]
```
Where ["table1. column1", "table2. column2", "key1", "key2"] means that "table1. column1" and "table2. column2" have the same meaning and may be redundant. 'key1' represents the Join key for column 'table1. column1', and 'key2' represents the Join key for column 'table2. column2'. The core function of association keys is to serve as the basis for data matching between different tables. By connecting rows with the same value, it enables us to cross the boundaries of the table and integrate scattered stored information into a meaningful and coherent data view. Here, you can determine whether there is data inconsistency between columns by using association keys.

Take a deep breath and think step by step to find the identify redundant columns.
'''
    # client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url="https://www.dmxapi.com/v1/")
    # client = OpenAI(api_key='sk-9e21d6fbb3e49fabc05a4869f136a72', base_url='https://api.deepseek.com')
    # response = client.chat.completions.create(
    #         # model=opt.model_name,
    #         model = "deepseek-reasoner",
    #         messages=[
    #             {"role": "system", "content": "You are a data science expert."},
    #             {"role": "user", "content": prompt}
    #         ],
    #         stream=False
    #     )
    # result = response.choices[0].message.content
    # print(f"LLM 输入：\n{prompt}")
    # assert 1==0
    # print(f"LLM 输出text：\n{result}")
    # match = re.search(r"```json\s*(.*?)\s*```", result, re.DOTALL)
    # redundant_columns_list = []
    # if match:
    #     json_str = match.group(1).strip()
    #     try:
    #         redundant_columns_list = json.loads(json_str)
    #     except Exception as e:
    #         print(e)
    redundant_columns_list = [
  ["frpm.`County Name`", "schools.County", "frpm.CDSCode", "schools.CDSCode"],
  ["frpm.`County Name`", "satscores.cname", "frpm.CDSCode", "satscores.cds"],
  ["schools.County", "satscores.cname", "schools.CDSCode", "satscores.cds"],
  ["frpm.`District Name`", "schools.District", "frpm.CDSCode", "schools.CDSCode"],
  ["frpm.`District Name`", "satscores.dname", "frpm.CDSCode", "satscores.cds"],
  ["schools.District", "satscores.dname", "schools.CDSCode", "satscores.cds"],
  ["frpm.`School Name`", "schools.School", "frpm.CDSCode", "schools.CDSCode"],
  ["frpm.`School Name`", "satscores.sname", "frpm.CDSCode", "satscores.cds"],
  ["schools.School", "satscores.sname", "schools.CDSCode", "satscores.cds"],
  ["frpm.`Charter School (Y/N)`", "schools.Charter", "frpm.CDSCode", "schools.CDSCode"],
  ["frpm.`Charter School Number`", "schools.CharterNum", "frpm.CDSCode", "schools.CDSCode"],
  ["frpm.`Charter Funding Type`", "schools.FundingType", "frpm.CDSCode", "schools.CDSCode"]
]
    print(f"LLM 输出json：\n{redundant_columns_list}")
    if redundant_columns_list:
        consistency_redundant_columns_list = []
        inconsistency_redundant_columns_list = []
        for redundant_item in redundant_columns_list:
            table1, column1 = redundant_item[0].split('.')
            table2, column2 = redundant_item[1].split('.')
            key1 = redundant_item[2].split('.')[1]
            key2 = redundant_item[3].split('.')[1]
            sql_statement = f'''SELECT
    {table1}.{key1},
    {table1}.{column1},
    {table2}.{key2},
    {table2}.{column2}
    FROM
        {table1} 
    JOIN
        {table2} ON {table1}.{key1} = {table2}.{key2}
    WHERE
        {table1}.{column1} <> {table2}.{column2}
    '''
            execute_result = db_model.execute_sql(sql_statement)
            if execute_result.empty: 
                consistency_redundant_columns_list.append(redundant_item)
            else:
                inconsistency_redundant_columns_list.append(redundant_item)
   
    print(f"一致性列：\n{consistency_redundant_columns_list}")
    print(f"不一致性列：\n{inconsistency_redundant_columns_list}")

    # 4. 最后，组合这些信息 
    # input schema 和question 描述直接在之前的arctic上面加
    # v1 这是直接加json格式的函数依赖关系
    database_information = "The following are the cardinality relationships between attributes. A many-to-one (N:1) relationship exists between attribute A and attribute B ([A, B]) if A functionally determines B (i.e., A → B), but B does not functionally determine A. In this case, each value of A can be associated with multiple values of B, but each value of B corresponds to exactly one value of A. A one-to-one (1:1) relationship exists between attribute A and attribute B if both A → B and B → A hold. This means each value of A corresponds to exactly one value of B, and vice versa. Any attribute pairs not mentioned are assumed to have a many-to-many (N:N) relationship. For example, the ratio between student ID and age is N:1, as there may be multiple students with the same age. \n" 
    for table in database_ratio_maps:
        database_information += f"In table `{table}`: \n"
        for key, value in database_ratio_maps[table].items():
            database_information += f"{key}: {value} \n"
    database_information += "Not all many-to-to (N:1) relationships are listed here. Whenever A is a primary key and B is any other attribute, the relationship between A and B is considered to be many-to-one by default.\nThese cardinality relationships will influence how you generate SQL queries. \n"

    # v2 加文本描述的函数依赖关系
    # database_information = "The following is the cardinality relationship between attributes that do not include a primary key. \n" + database_ratio_text
    # database_information += "\nThe proportion relationship involving the primary key is not listed. Because obviously, the proportional relationship between any other attribute and the primary key is many-to-one (N:1).\n"
    # database_information += "When the question involves these columns, you need to first consider these proportional relationships, which will affect how you write SQL.\n"

    database_information += "In addition, **there are some redundant columns here, but their stored data is consistent. You can use one of them freely.**\n"
    for consistency_redundant_columns in consistency_redundant_columns_list:
        database_information += f"{consistency_redundant_columns[0]} and {consistency_redundant_columns[1]}\n"
    
    database_information += "**There are also some redundant columns, but the data they store is inconsistent. When querying involving these columns, you need to carefully consider which column to use.**\n"
    for inconsistency_redundant_columns in inconsistency_redundant_columns_list:
        database_information += f"{inconsistency_redundant_columns[0]} and {inconsistency_redundant_columns[1]}\n"

    new_dataset = []
    with open(opt.schema_description_path, 'r', encoding='utf-8') as ref_f, open(opt.input_data_file, 'r', encoding='utf-8') as input_f:
        schema_question_pairs = json.load(ref_f)
        input_questions = json.load(input_f)
        for i, input_question in enumerate(input_questions):
            if input_question['db_id'] == opt.database:
                schema_question_pair = schema_question_pairs[i]
                # 改造一下input_seq
                input_prompt_1, input_prompt_2 = schema_question_pair['input_seq'].split('Instructions:')
                input_prompt = input_prompt_1 + "\nKnowledge:\n" + database_information + "\nInstructions:\n" + input_prompt_2
                schema_question_pair['input_seq'] = input_prompt
                schema_question_pair['knowledge'] = database_information
                schema_question_pair['fd_text'] = database_ratio_text
                schema_question_pair['consistency_redundant_columns'] = consistency_redundant_columns
                schema_question_pair['inconsistency_redundant_columns'] = inconsistency_redundant_columns
                new_dataset.append(schema_question_pair)
                
    with open(opt.output_data_file, 'w', encoding='utf-8') as save_f:
        save_f.write(json.dumps(new_dataset, indent=2, ensure_ascii=False))

    print(f"finish saving file to {opt.output_data_file}")




    # 加载模型

    # 生成sql

    # 保存
