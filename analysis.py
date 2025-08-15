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


def test_evaluate():
    from itertools import combinations, permutations

    def flexible_result_comparison(predicted, ground_truth, max_attempts=1000):
        """
        灵活比较两个SQL查询结果
        处理以下情况：
        1. predicted结果包含额外的列
        2. 列的顺序不同（即使列数相同）
        """
        if not predicted or not ground_truth:
            return len(predicted) == len(ground_truth)
        
        # 获取列数
        pred_cols = len(predicted[0])
        truth_cols = len(ground_truth[0])
        
        # 如果predicted的列数少于ground_truth，直接返回False
        if pred_cols < truth_cols:
            return False
        
        # 如果列数相同，需要考虑所有列的排列
        if pred_cols == truth_cols:
            attempts = 0
            for perm in permutations(range(pred_cols)):
                if attempts >= max_attempts:
                    break
                attempts += 1
                
                # 重新排列predicted的列
                reordered_predicted = []
                for row in predicted:
                    reordered_row = tuple(row[i] for i in perm)
                    reordered_predicted.append(reordered_row)
                
                # 比较重新排列后的结果
                if set(reordered_predicted) == set(ground_truth):
                    return True
            return False
        
        # 如果predicted有更多列，需要先选择列组合，再考虑排列
        attempts = 0
        for col_indices in combinations(range(pred_cols), truth_cols):
            if attempts >= max_attempts:
                break
                
            # 对选中的列考虑所有可能的排列
            for perm in permutations(range(truth_cols)):
                attempts += 1
                if attempts > max_attempts:
                    break
                    
                # 首先按col_indices提取列，然后按perm重新排列
                filtered_predicted = []
                for row in predicted:
                    # 先提取指定的列
                    selected_cols = [row[col_indices[i]] for i in range(truth_cols)]
                    # 再按排列重新组织
                    reordered_row = tuple(selected_cols[perm[i]] for i in range(truth_cols))
                    filtered_predicted.append(reordered_row)
                
                # 比较结果
                if set(filtered_predicted) == set(ground_truth):
                    return True
        
        return False

    # ===== 测试代码 =====
    print("=== 测试验证 ===")

    # 测试1: 列数相同，顺序不同
    print("测试1: 列数相同，顺序不同")
    predicted_1 = [('Alice', 25), ('Bob', 30)]  # name, age
    ground_truth_1 = [(25, 'Alice'), (30, 'Bob')]  # age, name
    result_1 = flexible_result_comparison(predicted_1, ground_truth_1)
    print(f"结果: {result_1} (应该是True)")

    # 测试2: predicted多列，包含正确的列但顺序不同  
    print("\n测试2: predicted多列，顺序不同")
    predicted_2 = [(1, 'Alice', 25), (2, 'Bob', 30)]  # id, name, age
    ground_truth_2 = [(25, 'Alice'), (30, 'Bob')]  # age, name
    result_2 = flexible_result_comparison(predicted_2, ground_truth_2)
    print(f"结果: {result_2} (应该是True)")

    # 测试3: 完全匹配
    print("\n测试3: 完全匹配")
    predicted_3 = [('Alice', 25), ('Bob', 30)]
    ground_truth_3 = [('Alice', 25), ('Bob', 30)]
    result_3 = flexible_result_comparison(predicted_3, ground_truth_3)
    print(f"结果: {result_3} (应该是True)")

    # 测试4: 真正不匹配的情况
    print("\n测试4: 真正不匹配")
    predicted_4 = [('Alice', 25), ('Bob', 30)]
    ground_truth_4 = [('Charlie', 35), ('David', 40)]
    result_4 = flexible_result_comparison(predicted_4, ground_truth_4)
    print(f"结果: {result_4} (应该是False)")

    # 额外测试: 验证排列逻辑
    print("\n=== 验证排列逻辑 ===")
    from itertools import permutations
    test_data = [('Alice', 25), ('Bob', 30)]
    print(f"原始数据: {test_data}")
    for i, perm in enumerate(permutations(range(2))):
        reordered = [tuple(row[j] for j in perm) for row in test_data]
        print(f"排列{i} {perm}: {reordered}")

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

# test_parse_sql()
# test_evaluate()


rows = [('a.lucero@realjourney.org', 'j.hernandez@realjourney.org', None), ('jfranco@ofy.org', 'bgillespie@ofy.org', None), ('a.lucero@realjourney.org', 'a.ramirez@realjourney.org', None), ('tallen@fortuneschool.us', 'bbensen@fortuneschool.us', None)]
column_none_counts = {0: 0, 1: 0, 2: 4}

content_3 = f"""Based on your previously generated SQL:
the execution returned 0 rows, with the first five rows being: 
{rows[:5]}
For the following columns (indexed starting from 1), the number of None values are respectively: 
""" + '.\n'.join([f'column {col+1} have {count} None value' for col, count in column_none_counts.items()]) + """

You can revise the SQL based on the user's question and the execution results, adding NOT NULL checks to the relevant columns when **necessary** to avoid None values.
"""

print(content_3)