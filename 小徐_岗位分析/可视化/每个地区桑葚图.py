from pyecharts.charts import Sankey
from pyecharts import options as opts
import pandas as pd
from collections import defaultdict

df = pd.read_csv(r"D:\360MoveData\Users\30648\Desktop\58同城数据\广州\广州1.csv")
welfare_fields = ["五险一金", "包住", "包吃", "年底双薪", "周末双休", "交通补助", "加班补助", "饭补", "话补", "房补"]

link_counter = defaultdict(int)
for _, row in df.iterrows():
    job = row["类别"]
    for welfare in welfare_fields:
        if str(row[welfare]).strip() not in ["无", "nan", "", None]:
            link_counter[(job, welfare)] += 1

nodes = set()
links = []
for (job, welfare), count in link_counter.items():
    nodes.add(job)
    nodes.add(welfare)
    links.append({"source": job, "target": welfare, "value": count})
nodes = [{"name": n} for n in nodes]

sankey = (
    Sankey(init_opts=opts.InitOpts(width="1000px", height="800px"))
    .add(
        "岗位与福利关系图",
        nodes=nodes,
        links=links,
        linestyle_opt=opts.LineStyleOpts(opacity=0.3, curve=0.5, color="source"),
        label_opts=opts.LabelOpts(position="right"),
    )
    .set_global_opts(title_opts=opts.TitleOpts(title="职位类别与福利项的桑葚图"))
)

sankey.render("高级桑葚图.html")  # 输出文件
