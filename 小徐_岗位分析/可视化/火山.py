import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font",family='YouYuan')
# 读取数据（假设数据已经载入为 df）
# 在这里替换为你自己的数据文件路径
df = pd.read_csv(r"D:\360MoveData\Users\30648\Desktop\58同城数据\广州\广州1.csv")

# 计算薪资范围的差异（假设数据格式为 '薪资' 列，薪资范围如 '5000-10000'）
df['薪资范围'] = df['薪资'].str.split('-').apply(lambda x: [int(i) for i in x] if len(x) == 2 else [None, None])
df['最低薪资'] = df['薪资范围'].apply(lambda x: x[0] if x else None)
df['最高薪资'] = df['薪资范围'].apply(lambda x: x[1] if x else None)

# 计算薪资差异的对数作为 X 轴
df['薪资差异'] = np.log(df['最高薪资'].fillna(1) / df['最低薪资'].fillna(1))

# 计算显著性（这里我们使用随机数据来模拟显著性，实际应用中应使用统计测试计算 p 值）
df['显著性'] = np.random.uniform(0, 1, size=len(df))  # 假设显著性为随机数据
df['p值负对数'] = -np.log10(df['显著性'])

# 创建火山图
plt.figure(figsize=(10, 6))

# 绘制散点图
plt.scatter(df['薪资差异'], df['p值负对数'], c=df['显著性'], cmap='coolwarm', alpha=0.7)

# 添加标题和标签
plt.title('薪资差异显著性火山图', fontsize=16)
plt.xlabel('薪资差异 (对数)', fontsize=14)
plt.ylabel('显著性 (-log10(p值))', fontsize=14)

# 添加显著性阈值线（例如 p < 0.05）
plt.axhline(y=-np.log10(0.05), color='gray', linestyle='--', label='显著性阈值 (p<0.05)')

# 添加图例
plt.legend()

# 显示图形
plt.show()
