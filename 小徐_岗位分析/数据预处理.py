import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.rc("font",family='YouYuan')

# 1. 读取 CSV 文件（注意文件路径的反斜杠需要转义，或者使用原始字符串 r""）
file_path = r"D:\360MoveData\Users\30648\Desktop\58同城数据\北京\北京.csv"
df = pd.read_csv(file_path, encoding='gbk', dtype=str)  # 强制所有列为字符串，防止空值变 NaN

# 2. 将所有空白单元格（包括 NaN、空字符串、仅空格）替换为 "无"
df.replace(to_replace=[None, np.nan, '', ' '], value='无', inplace=True)
df = df.applymap(lambda x: '无' if str(x).strip() == '' else x)

# 3. （可选）去除重复行
df.drop_duplicates(inplace=True)

# 4. （可选）清洗“薪资”字段
def clean_salary(salary):
    # 如果薪资为“面议”，则返回 None，表示删除该行数据
    if salary == '面议':
        return None
    
    # 去除所有中文字符
    salary = re.sub(r'[\u4e00-\u9fa5]', '', salary)
    
    # 提取数字
    nums = re.findall(r'\d+', salary)
    
    # 返回提取到的数字字符串，若没有提取到数字则返回 None
    return '-'.join(nums) if nums else None

if '薪资' in df.columns:
    df['薪资'] = df['薪资'].apply(clean_salary)

# 删除薪资为 None 的行
df = df[df['薪资'].notnull()]
# 清洗薪资字段并计算平均薪资
def calculate_average_salary(salary):
    if salary == '无' or salary is None:
        return np.nan
    try:
        # 找到数字部分
        nums = re.findall(r'\d+', salary)
        # 如果找到数字，计算平均
        if len(nums) == 2:
            return (int(nums[0]) + int(nums[1])) / 2
    except Exception as e:
        pass
    return np.nan

df['平均薪资'] = df['薪资'].apply(calculate_average_salary)

# 5. 保存清洗后的数据
output_path = r"D:\360MoveData\Users\30648\Desktop\58同城数据\北京\北京1.csv"
df.to_csv(output_path, index=False, encoding='utf-8-sig')

# 输出数据的信息
print(df.info())
# 1. 按地区分类，计算样本数量和平均薪资
district_stats = df.groupby('地区').agg(样本数量=('地区', 'size'), 平均薪资=('平均薪资', 'mean')).reset_index()

# 2. 绘制各个区样本数量的地图（需要地图的外部库，比如 geopandas）


# 3. 计算福利中“无”的比例
import matplotlib.pyplot as plt

benefits_columns = ['五险一金', '包住', '包吃', '年底双薪', '周末双休', 
                    '交通补助', '加班补助', '饭补', '话补', '房补']

# 计算每个福利项“无”占比
benefits_ratio = {col: (df[col] == '无').mean() for col in benefits_columns}
# 计算1 - benefits_ratio
benefits_ratio_complement = {col: 1 - value for col, value in benefits_ratio.items()}

# 设置颜色列表

# 设置子图数量（这里10个福利项所以10个子图）
fig, axes = plt.subplots(2, 5, figsize=(18, 8))  # 创建2行5列的子图
axes = axes.flatten()  # 展平 axes 数组，方便迭代
colors =  ['#FDEEF0',
    '#FADBDF',  # 非常浅的粉色（接近白色）
    '#F5B9C2',  # 浅粉色
    '#F6BDC5',  # 柔和的粉色
    '#F3ABB6',  # 深一点的粉色
    '#EE8594',  # 深粉色
    '#FAD1E7',  # 更深的粉色
    '#F8BEDE',  # 深红色（可选，用于渐变的末端）
    '#F7B5D9',  # 深红色（可选，用于渐变的末端）
    '#F6A9D2',  # 深红色（可选，用于渐变的末端）
    '#F285C0'   # 深红色（可选，用于渐变的末端）
]
# 绘制每个福利项的比例图
for i, (benefit, ratio) in enumerate(benefits_ratio_complement.items()):
    axes[i].barh([benefit], [ratio], color=colors[i])  # 使用颜色列表中的颜色
    axes[i].set_xlim(0, 1)  # 设置x轴从0到1
    axes[i].set_title(f'{benefit}：{ratio:.2%}')  # 显示标题并以百分比格式显示比例
    axes[i].set_xlabel('比例')
    axes[i].set_ylabel('')

# 调整布局，使得图表不重叠
plt.tight_layout()
plt.suptitle('福利中“无”占总体样本数量的比例', fontsize=16, y=1.05)
plt.show()

# 4. 计算各个类别的数量，找出数量最多的5种职业以及他们的平均薪资
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 假设 df 是你的数据框，并且已经定义了 top_jobs
top_jobs = df.groupby('类别').agg(数量=('类别', 'size'), 平均薪资=('平均薪资', 'mean')).nlargest(10, '数量').reset_index()

# 设置颜色列表
colors =  ['#FDEEF0',
    '#FADBDF',  # 非常浅的粉色（接近白色）
    '#F5B9C2',  # 浅粉色
    '#F6BDC5',  # 柔和的粉色
    '#F3ABB6',  # 深一点的粉色
    '#EE8594',  # 深粉色
    '#FAD1E7',  # 更深的粉色
    '#F8BEDE',  # 深红色（可选，用于渐变的末端）
    '#F7B5D9',  # 深红色（可选，用于渐变的末端）
    '#F6A9D2',  # 深红色（可选，用于渐变的末端）
    '#F285C0'   # 深红色（可选，用于渐变的末端）
]

# 创建图形和双坐标轴
fig, ax1 = plt.subplots(figsize=(12, 6))

# 绘制柱状图（数量）
sns.barplot(data=top_jobs, x='类别', y='数量', palette=colors[:len(top_jobs)], alpha=0.7, ax=ax1)
ax1.set_xlabel('类别')
ax1.set_ylabel('数量')
ax1.tick_params(axis='y')
ax1.set_title('数量最多的10种职业')

# 创建第二个坐标轴（平均薪资）
ax2 = ax1.twinx()
ax2.set_ylabel('平均薪资',)
ax2.tick_params(axis='y')

# 绘制折线图（平均薪资）
ax2.plot(top_jobs['类别'], top_jobs['平均薪资'], color="#ED5EAB", marker='o', linestyle='-', linewidth=2, markersize=8)

# 设置x轴刻度旋转
plt.xticks(rotation=45)

# 显示图形
plt.show()
#词云图

from wordcloud import WordCloud
# 计算数量最多的10种职业
top10_jobs = df['类别'].value_counts().nlargest(15)

# 创建词云图, 指定中文字体
wordcloud = WordCloud(
    font_path='C:\WINDOWS\FONTS\SIMYOU.TTF',  
    width=800,
    height=400,
    background_color='white',
    colormap = 'Reds',
    max_words=15,  
    random_state=42).generate_from_frequencies(top10_jobs)

# 显示词云图
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # 不显示坐标轴
plt.title('数量最多的15种职业词云图', fontsize=16)
plt.show()
#散点图
# 找出“类别”中样本数量最多的前10项
top_categories = df['类别'].value_counts().nlargest(10).index.tolist()

# 筛选出这些类别的数据
top_data = df[df['类别'].isin(top_categories)]

# 创建箱型图
# 选前 10 类别
top_categories = df['类别'].value_counts().nlargest(10).index.tolist()
top_data = df[df['类别'].isin(top_categories)]

# 设置蓝色调色板
blue_palette = sns.color_palette( ['#FDEEF0',
    '#FADBDF',  # 非常浅的粉色（接近白色）
    '#F5B9C2',  # 浅粉色
    '#F6BDC5',  # 柔和的粉色
    '#F3ABB6',  # 深一点的粉色
    '#EE8594',  # 深粉色
    '#FAD1E7',  # 更深的粉色
    '#F8BEDE',  # 深红色（可选，用于渐变的末端）
    '#F7B5D9',  # 深红色（可选，用于渐变的末端）
    '#F6A9D2',  # 深红色（可选，用于渐变的末端）
    '#F285C0'   # 深红色（可选，用于渐变的末端）
], 10)

# 绘图
plt.figure(figsize=(12, 8))
ax = sns.boxplot(x='类别', y='平均薪资', data=top_data, palette=blue_palette)

# 图形设置
plt.title('样本数量最多的10个职业的平均薪资分布（箱型图）', fontsize=14)
plt.xlabel('职业', fontsize=10)
plt.ylabel('平均薪资（元）', fontsize=10)
plt.xticks(rotation=45)

# 关闭背景网格线
ax.grid(False)

# 布局优化 + 显示图像
plt.tight_layout()
plt.show()

