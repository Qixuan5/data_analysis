import pandas as pd
import pyecharts.options as opts
from pyecharts.charts import Bar3D

# 读取数据
file_path = r"D:\360MoveData\Users\30648\Desktop\58同城数据\深圳\深圳1.csv"
df = pd.read_csv(file_path)
df['平均薪资'] = pd.to_numeric(df['平均薪资'], errors='coerce')
df = df.dropna(subset=['平均薪资'])

# Top5 职业
top_jobs = df['类别'].value_counts().nlargest(5).index.tolist()

# 所有地区（包括那些没有 top5 职业的）
all_districts = sorted(df['地区'].dropna().unique().tolist())

# 构建 [地区, 职业, 平均薪资, 人数] 数据矩阵
data = []
for district in all_districts:
    for job in top_jobs:
        filtered = df[(df['地区'] == district) & (df['类别'] == job)]
        if not filtered.empty:
            salary = filtered['平均薪资'].mean()
            count = filtered.shape[0]  # 人数
        else:
            salary = 0  # 或 None, 如果想让图中没有柱子
            count = 0
        data.append([district, job, round(salary, 2), count])

# 创建 Bar3D 图
(
    Bar3D()
    .add(
        series_name="平均薪资",
        data=[(d[2], d[3], d[0]) for d in data],  # 将 [平均薪资, 人数, 地区] 作为数据
        xaxis3d_opts=opts.Axis3DOpts(
            type_="value",
            name="平均薪资",
            axislabel_opts=opts.LabelOpts(interval=0, rotate=45, font_size=10)
        ),
        yaxis3d_opts=opts.Axis3DOpts(
            type_="category",
            data=top_jobs,
            axislabel_opts=opts.LabelOpts(font_size=10)
        ),
        zaxis3d_opts=opts.Axis3DOpts(
            type_="value",
            name="人数",
        ),
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="各区热门职业薪资与人数 3D 可视化"),
        visualmap_opts=opts.VisualMapOpts(
            max_=max([d[2] for d in data]),  # 最大薪资值
            range_color=['#FDEEF0', '#FADBDF', '#F5B9C2', '#F6BDC5', '#F3ABB6', '#EE8594'],
        ),
        toolbox_opts=opts.ToolboxOpts(is_show=True)
    )
    .render("深圳_3D_职业薪资与人数.html")
)
