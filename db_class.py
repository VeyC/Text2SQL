import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from itertools import combinations

import sqlite3


class Database:
    def __init__(self, db_path, table_name=None):
        """
        初始化DATABASE算法类
        Args:
            db_path: SQLite数据库路径
            table_name: 要分析的表名（如果为None，会列出所有表让用户选择）
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.table_name = table_name
        self.data = None
        
        if table_name is None:
            self.list_tables()
        else:
            self.load_data()
    
    def list_tables(self):
        """列出数据库中的所有表"""
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        tables = pd.read_sql_query(query, self.conn)
        print("数据库中的表:")
        for i, table in enumerate(tables['name']):
            print(f"{i+1}. {table}")
        return tables['name'].tolist()
    
    def get_database_ddls(self) -> str:
        """得到数据库的DDL语句"""
        query = "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        # 仅获取用户表（排除 sqlite 内部表）
        ddl = pd.read_sql_query(query, self.conn)
        full_ddl = "\n\n".join(ddl['sql'].tolist())
        return full_ddl
    
    def execute_sql(self, query):
        res = pd.read_sql_query(query, self.conn)
        return res
    
    def set_table(self, table_name):
        """设置要分析的表"""
        self.table_name = table_name
        self.load_data()
    
    def load_data(self):
        """从SQLite数据库加载数据"""
        if self.table_name is None:
            print("请先设置表名")
            return
        
        try:
            # 加载数据
            query = f"SELECT * FROM {self.table_name}"
            self.data = pd.read_sql_query(query, self.conn)
            print(f"成功加载表 '{self.table_name}'，数据形状: {self.data.shape}")
            print(f"列名: {list(self.data.columns)}")
            print(f"前5行数据:")
            print(self.data.head())
        except Exception as e:
            print(f"加载数据时出错: {e}")
    
    def get_table_schema(self):
        """获取表的结构信息"""
        if self.table_name is None:
            print("请先设置表名")
            return
        
        query = f"PRAGMA table_info({self.table_name})"
        schema = pd.read_sql_query(query, self.conn)
        print(f"表 '{self.table_name}' 的结构:")
        print(schema)
        return schema
    
    def get_table_attrs(self, table_name):
        query = f"PRAGMA table_info({table_name})"
        schema = pd.read_sql_query(query, self.conn)
        # 转换为字典结构
        schema_dict = {}
        for _, row in schema.iterrows():
            column_name = row['name']
            schema_dict[column_name] = {
                'type': row['type'],
                'notnull': bool(row['notnull']),
                'dflt_value': row['dflt_value'],
                'pk': bool(row['pk'])
            }
        return schema_dict
    
    def compute_entropy(self, LHS, RHS):
        """
        计算给定LHS和RHS的熵
        
        Args:
            LHS: 左侧属性列表
            RHS: 右侧属性
        
        Returns:
            熵值
        """
        if self.data is None:
            print("数据未加载")
            return float('inf')
        
        try:
            # 将数据转换为字符串以处理不同数据类型
            grouped_data = self.data.copy()
            for col in LHS + [RHS]:
                grouped_data[col] = grouped_data[col].astype(str)
            
            tmp = grouped_data.groupby(LHS)[RHS].nunique()
            entropy = (tmp > 1).sum()
            return entropy
        except Exception as e:
            print(f"计算熵时出错: {e}")
            return float('inf')
    
    def find_functional_dependencies(self, max_lhs_size=None):
        """
        使用TANE算法贪婪地找到函数依赖关系
        
        Args:
            max_lhs_size: LHS的最大大小限制（None表示无限制）
        
        Returns:
            函数依赖列表
        """
        if self.data is None:
            print("数据未加载，无法分析函数依赖")
            return []
        
        FD_list = []
        columns = list(self.data.columns)
        
        # 如果没有指定最大LHS大小，则使用列数
        if max_lhs_size is None:
            max_lhs_size = len(columns)
        
        print(f"开始分析函数依赖，最大LHS大小: {max_lhs_size}")
        
        # 从大小为2的组合开始，逐步增加到max_lhs_size+1
        for r in range(2, min(max_lhs_size + 2, len(columns) + 1)):
            print(f"正在分析大小为 {r} 的属性组合...")
            
            for comb in combinations(columns, r):
                for RHS in comb:
                    LHS = [col for col in comb if col != RHS]
                    
                    # 条件1: 检查是否已有更小的LHS能推导出相同的RHS
                    cond_1 = [r_t == RHS and len(set(LHS).intersection(set(l_t))) == len(l_t) 
                             for l_t, r_t in FD_list]
                    
                    # 条件2: 检查当前LHS是否包含已存在FD的所有属性
                    cond_2 = [set(l_t + [r_t]).intersection(set(LHS)) == set(l_t + [r_t]) 
                             for l_t, r_t in FD_list]
                    
                    if sum(cond_1) == 0 and sum(cond_2) == 0:
                        entropy = self.compute_entropy(LHS, RHS)
                        if entropy == 0:
                            FD_list.append([LHS, RHS])
                            print(f"发现函数依赖: {' '.join(LHS)} -> {RHS}")
        
        return FD_list
    
    def format_functional_dependencies(self, FD_list):
        """
        格式化函数依赖输出
        
        Args:
            FD_list: 函数依赖列表
        
        Returns:
            格式化的函数依赖字符串列表
        """
        formatted_fds = []
        for lhs, rhs in FD_list:
            lhs_str = ''.join(lhs) if isinstance(lhs, list) else lhs
            formatted_fds.append(f"{lhs_str} -> {rhs}")
        return formatted_fds
    
    def analyze_specific_table(self, table_name, max_lhs_size=3):
        """
        分析指定表的函数依赖
        
        Args:
            table_name: 表名
            max_lhs_size: LHS最大大小
        """
        print(f"\n=== 分析表: {table_name} ===")
        self.set_table(table_name)
        
        if self.data is not None:
            # 显示表结构
            print(f"\n表结构信息:")
            self.get_table_schema()
            
            # 寻找函数依赖
            print(f"\n开始寻找函数依赖...")
            fd_list = self.find_functional_dependencies(max_lhs_size)
            
            # 输出结果
            if fd_list:
                print(f"\n发现的函数依赖关系:")
                formatted_fds = self.format_functional_dependencies(fd_list)
                for i, fd in enumerate(formatted_fds, 1):
                    print(f"{i}. {fd}")
                print(f"\n总共发现 {len(fd_list)} 个函数依赖关系")
            else:
                print("\n未发现函数依赖关系")
            
            return fd_list
    

    def analyze_column_distribution(self, table_name=None):
        """
        分析表中所有列的值分布情况
        
        Args:
            table_name: 表名（如果为None，使用当前设置的表）
        
        Returns:
            dict: 每列的分布统计信息
                - unique_count: 不同值的个数
                - top_5_values: 出现次数最高的5个值及其频次
                - total_count: 总记录数
                - null_count: 空值数量
        """
        if table_name:
            temp_table = self.table_name
            self.set_table(table_name)
        
        if self.data is None:
            print("数据未加载，无法分析分布")
            return {}
        
        distribution_stats = {}
        
        print(f"\n=== 分析表 '{self.table_name}' 的列分布情况 ===")
        
        for column in self.data.columns:
            try:
                # 计算基本统计信息
                total_count = len(self.data)
                null_count = self.data[column].isnull().sum()
                non_null_data = self.data[column].dropna()
                
                # 计算不同值的个数
                unique_count = non_null_data.nunique()
                
                # 获取值的频次统计
                value_counts = non_null_data.value_counts()
                
                # 获取出现次数最高的5个值
                top_5_values = value_counts.head(5).to_dict()
                top_5_list = [(value, count) for value, count in top_5_values.items()]
                                
                # 存储统计信息
                distribution_stats[column] = {
                    'unique_count': unique_count,
                    'top_5_values': top_5_list,
                    'null_count': null_count
                }
               
                
            except Exception as e:
                print(f"分析列 '{column}' 时出错: {e}")
                distribution_stats[column] = {
                    'error': str(e),
                    'unique_count': 0,
                    'top_5_values': [],
                    'total_count': 0,
                    'null_count': 0
                }
        
        # 恢复原来的表设置
        if table_name and temp_table:
            self.table_name = temp_table
            self.load_data()
        
        return distribution_stats


    def close(self):
        """关闭数据库连接"""
        self.conn.close()
 