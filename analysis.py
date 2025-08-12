# 用于分析数据
import json
import sqlglot


def test_parse_sql():
    sql = "SELECT T1.`District Code` FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.City = 'Fresno' AND T2.Magnet = 0"

    expression = sqlglot.parse_one(sql, dialect='sqlite')

    # 提取所有列（包括SELECT、WHERE、JOIN等位置）
    columns = expression.find_all(sqlglot.exp.Column)
    columns_used = set(str(col) for col in columns)
    print("所有涉及的列：", columns_used)

# if __name__ == '__main__':
#     path = '../output/bird/dev_bird.json'
#     with open(path, 'r', encoding='utf-8') as f:
#         datas = json.load(f)
#     for i, data in enumerate(datas):
#         # if 'Charter Funding Type' in data['output_seq']:
#         if 'NOT NULL' in data['output_seq']:
#             print('---------'*5)
#             print(i, data['question'])
#             print(data['output_seq'])

test_parse_sql()
