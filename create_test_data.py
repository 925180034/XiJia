"""
创建测试数据文件
"""
import os
import pandas as pd

def create_test_data():
    """创建测试数据文件"""
    print("创建测试数据目录...")
    os.makedirs("data", exist_ok=True)
    
    print("创建源数据字典...")
    # 创建源数据字典
    source_data = {
        '表名': ['T_XGXT_T_SS_ZNCQ_XSGSQK'] * 5,
        '表描述': ['归宿情况'] * 5,
        '字段名': ['XSBH', 'XM', 'WID', 'QKLX', 'LOGIN_TIME'],
        '字段描述': ['学生编号', '姓名', 'WID', '情况类型', '登录时间'],
        '字段类型': ['VARCHAR', 'VARCHAR', 'VARCHAR', 'INT', 'DATETIME']
    }

    source_df = pd.DataFrame(source_data)
    source_file = 'data/源数据字典.xlsx'
    source_df.to_excel(source_file, index=False)
    print(f"源数据字典已保存: {source_file}")

    print("创建目标数据字典...")
    # 创建目标数据字典
    target_data = {
        '表名': ['T_RKJXXXHPT_T_CXJX_BKSXKXX'] * 5,
        '表描述': ['本科生学生信息'] * 5,
        '字段名': ['XH', 'XM', 'WID', 'XDRSDM', 'FB_TIME'],
        '字段描述': ['学号', '姓名', 'WID', '下达人身份码', 'FB_TIME'],
        '字段类型': ['VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'DATETIME']
    }

    target_df = pd.DataFrame(target_data)
    target_file = 'data/项目匹配字典.xlsx'
    target_df.to_excel(target_file, index=False)
    print(f"目标数据字典已保存: {target_file}")
    
    print("测试数据创建完成！")

if __name__ == "__main__":
    create_test_data()