import pandas as pd
from pyecharts.charts import Sankey
from pyecharts import options as opts

# ✅ 读取数据
df = pd.read_csv(r"D:\360MoveData\Users\30648\Desktop\58同城数据\深圳\深圳1.csv")
df.columns = [col.strip() for col in df.columns]

# ✅ 福利字段
welfare_fields = [
    "五险一金", "包住", "包吃", "年底双薪", "周末双休",
    "交通补助", "加班补助", "饭补", "话补", "房补"
]

# ✅ 薪资字段清洗
df["平均薪资"] = (
    df["平均薪资"]
    .astype(str)
    .str.replace("（.*?）", "", regex=True)
    .str.replace(",", "")
    .str.extract(r"(\d+)")[0]
    .astype(float)
)

# ✅ 福利数量
df["福利数量"] = df[welfare_fields].apply(lambda row: sum(v.strip() != "无" for v in row), axis=1)

# ✅ 每个地区取前3高频岗位类别（稳定写法）
top_jobs = (
    df.groupby("地区")["类别"]
    .apply(lambda x: x.value_counts().nlargest(3).index.tolist())
    .explode()
    .reset_index()
)
df = df.merge(top_jobs, on=["地区", "类别"])

# ✅ 工资段函数
def salary_range(salary):
    if salary >= 20000:
        return "￥20K+"
    elif salary >= 15000:
        return "￥15K-20K"
    elif salary >= 10000:
        return "￥10K-15K"
    else:
        return "￥10K以下"

# ✅ 福利等级函数
def welfare_level(count):
    if count >= 6:
        return "福利优秀"
    elif count >= 3:
        return "福利良好"
    else:
        return "福利一般"

# ✅ 更清晰的颜色映射
city_color_map = {
    "北京": "#FF6F61",
    "上海": "#6B5B95",
    "广州": "#88B04B",
    "深圳": "#009B77"
}
salary_color_map = {
    "￥20K+": "#045275",
    "￥15K-20K": "#2E8B57",
    "￥10K-15K": "#5DA5DA",
    "￥10K以下": "#B0C4DE"
}
welfare_color_map = {
    "福利优秀": "#F28E2B",
    "福利良好": "#FFBF00",
    "福利一般": "#CFCFC4"
}

# ✅ 构建节点和链接
from collections import defaultdict

nodes_dict = {}
link_counter = defaultdict(int)

for _, row in df.iterrows():
    city = row["地区"].strip()
    job = row["类别"].strip()
    salary = salary_range(row["平均薪资"])
    welfare = welfare_level(row["福利数量"])

    # 记录出现次数作为线宽
    link_counter[(city, job)] += 1
    link_counter[(job, salary)] += 1
    link_counter[(salary, welfare)] += 1

    # 节点收集 + 配色
    for name in [city, job, salary, welfare]:
        if name not in nodes_dict:
            if name in city_color_map:
                color = city_color_map[name]
            elif name in salary_color_map:
                color = salary_color_map[name]
            elif name in welfare_color_map:
                color = welfare_color_map[name]
            else:
                color = "#CCCCCC"
            nodes_dict[name] = {"name": name, "itemStyle": {"color": color}}

# 构造 link 列表（带权重）
links = [
    {"source": source, "target": target, "value": count}
    for (source, target), count in link_counter.items()
]

nodes = list(nodes_dict.values())

# ✅ 绘制桑基图（数据可读性优先）
sankey = (
    Sankey(init_opts=opts.InitOpts(width="1300px", height="700px", bg_color="#FFFFFF"))
    .add(
        series_name="北上广深招聘流向",
        nodes=nodes,
        links=links,
        linestyle_opt=opts.LineStyleOpts(opacity=0.4, curve=0.4, color="source"),
        label_opts=opts.LabelOpts(
            position="right",
            font_size=14,
            font_weight="bold",
            color="#333"
        ),
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="北上广深 岗位 → 薪资 → 福利 桑基图",
            title_textstyle_opts=opts.TextStyleOpts(font_size=20, font_weight="bold"),
            pos_left="center"
        ),
        toolbox_opts=opts.ToolboxOpts(
            is_show=True,
            feature=opts.ToolBoxFeatureOpts(
                save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(is_show=True),
                restore=opts.ToolBoxFeatureRestoreOpts(is_show=True)
            )
        ),
        tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{b}")
    )
    .render("北上广深_岗位_薪资_福利_桑基图.html")
)

print("✅ 成功生成：北上广深_岗位_薪资_福利_桑基图.html")
