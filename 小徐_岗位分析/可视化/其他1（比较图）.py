from pyecharts.charts import PictorialBar, Grid
from pyecharts import options as opts
from pyecharts.globals import SymbolType

# 数据
cities = ["上海", "北京", "广州", "深圳"]
salary = [9053.04, 8071.86, 7422.91, 9838.24]
welfare = [6.75, 6.97, 6.95, 7.11]
colors = ["#5470C6", "#91CC75", "#FAC858", "#EE6666"]

# 平均薪资图
bar1 = (
    PictorialBar()
    .add_xaxis(cities)
    .add_yaxis(
        "平均薪资",
        salary,
        label_opts=opts.LabelOpts(is_show=False),
        symbol_size=18,
        symbol_repeat="fixed",
        symbol_offset=[0, 0],
        is_symbol_clip=True,
        symbol=SymbolType.ROUND_RECT,
        color=colors,
    )
    .reversal_axis()
    .set_global_opts(
        title_opts=opts.TitleOpts(title="北上广深 - 平均薪资"),
        xaxis_opts=opts.AxisOpts(is_show=False),
        yaxis_opts=opts.AxisOpts(is_show=True),
    )
)

# 平均福利图
bar2 = (
    PictorialBar()
    .add_xaxis(cities)
    .add_yaxis(
        "平均福利数",
        welfare,
        label_opts=opts.LabelOpts(is_show=False),
        symbol_size=18,
        symbol_repeat="fixed",
        symbol_offset=[0, 0],
        is_symbol_clip=True,
        symbol=SymbolType.ROUND_RECT,
        color=colors,
    )
    .reversal_axis()
    .set_global_opts(
        title_opts=opts.TitleOpts(title="北上广深 - 平均福利数"),
        xaxis_opts=opts.AxisOpts(is_show=False),
        yaxis_opts=opts.AxisOpts(is_show=True),
    )
)

# 合并图表为上下两个图
grid = (
    Grid()
    .add(bar1, grid_opts=opts.GridOpts(pos_bottom="55%"))
    .add(bar2, grid_opts=opts.GridOpts(pos_top="60%"))
    .render("salary_welfare_pictorial.html")
)