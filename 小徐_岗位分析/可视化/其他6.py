from pyecharts.charts import Graph
from pyecharts import options as opts
import pandas as pd
import numpy as np
import re

# ✅ 正确的文件编码（不要用 utf-8）
df = pd.read_csv(r"D:\360MoveData\Users\30648\Desktop\58同城数据\深圳\深圳1.csv")


# ✅ 字段配置
attribute_fields = [
    "地区", "五险一金", "包住", "包吃", "年底双薪", "周末双休",
    "交通补助", "加班补助", "饭补", "话补", "房补"
]

# ✅ 获取最多的10个类别
top_10_categories = df['类别'].value_counts().nlargest(10).index.tolist()
df = df[df['类别'].isin(top_10_categories)]

# ✅ 分类设置：前10类别 + 地区 + 福利
categories = [{"name": cat} for cat in top_10_categories]
categories.append({"name": "地区"})
categories.append({"name": "福利"})

# ✅ 分类对应编号
category_map = {cat: i for i, cat in enumerate(top_10_categories)}
category_map["地区"] = 10
category_map["福利"] = 11

# ✅ 收集节点名
welfare_set = set()
district_set = set()

for _, row in df.iterrows():
    for field in attribute_fields:
        value = str(row.get(field, "无")).strip()
        if value != "无":
            if field == "地区":
                district_set.add(value)
            else:
                welfare_set.add(field)

# ✅ 构造节点列表（福利 → 地区 → 类别），确保 circular 布局紧邻
# ✅ 构造节点列表（福利 → 地区 → 类别），确保 circular 布局紧邻
nodes = []
node_names = set()

# 统计每个福利出现次数
welfare_counts = {w: 0 for w in welfare_set}
district_counts = {d: 0 for d in district_set}
for _, row in df.iterrows():
    for field in attribute_fields:
        value = str(row.get(field, "无")).strip()
        if value != "无":
            if field == "地区":
                district_counts[value] += 1
            else:
                welfare_counts[field] += 1

# 添加福利节点（大小按出现次数）
for w in sorted(welfare_set):
    count = welfare_counts[w]
    nodes.append({
        "name": w,
        "symbolSize": max(10, min(count * 0.8, 30)),  # 控制范围 [10, 30]
        "category": category_map["福利"]
    })
    node_names.add(w)

# 添加地区节点（大小按出现次数）
for d in sorted(district_set):
    count = district_counts[d]
    nodes.append({
        "name": d,
        "symbolSize": max(10, min(count * 0.8, 30)),
        "category": category_map["地区"]
    })
    node_names.add(d)

# 统计类别数量
category_counts = df['类别'].value_counts()

# 添加类别节点（大小按数量）
for cat in top_10_categories:
    count = category_counts[cat]
    nodes.append({
        "name": cat,
        "symbolSize": max(20, min(count * 0.5, 50)),  # 控制范围 [20, 50]
        "category": category_map[cat]
    })
    node_names.add(cat)


# ✅ 构建连接线：源是福利/地区，目标是类别；颜色跟随类别
links = []
for _, row in df.iterrows():
    category = str(row["类别"])
    for field in attribute_fields:
        value = str(row.get(field, "无")).strip()
        if value != "无":
            node_label = value if field == "地区" else field
            if node_label in node_names:
                links.append({
                    "source": node_label,
                    "target": category,
                    "lineStyle": {"color": None}  # 自动跟随 target 节点颜色
                })

# ✅ 构建图表
graph = (
    Graph(init_opts=opts.InitOpts(width="1400px", height="800px"))
    .add(
        series_name="广州招聘图",
        nodes=nodes,
        links=links,
        categories=categories,
        layout="circular",
        is_rotate_label=True,
        linestyle_opts=opts.LineStyleOpts(color="target", curve=0.3),
        label_opts=opts.LabelOpts(position="right"),
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="广州招聘前十职业与地区/福利关系图"),
        legend_opts=opts.LegendOpts(orient="vertical", pos_left="2%", pos_top="20%"),
    )
    .render("广州招聘_前十职业_福利地区_分组图.html")
)
