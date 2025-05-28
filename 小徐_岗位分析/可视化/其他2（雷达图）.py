from pyecharts import options as opts
from pyecharts.charts import Radar

# 北京的数据
value_bj = [
    [1, 2, 1, 1, 1, 1, 1],  # 五险一金排名, 包吃包住排名, 年底双薪排名, 周末双休排名, 加班补助排名, 其他补助排名
]

# 上海的数据
value_sh = [
    [2, 1, 2, 2, 2, 2, 2],  # 五险一金排名, 包吃包住排名, 年底双薪排名, 周末双休排名, 加班补助排名, 其他补助排名
]

# 广州的数据
value_gz = [
    [4, 4, 4, 4, 4, 4, 4],  # 五险一金排名, 包吃包住排名, 年底双薪排名, 周末双休排名, 加班补助排名, 其他补助排名
]

# 深圳的数据
value_sz = [
    [3, 3, 3, 3, 3, 3, 3],  # 五险一金排名, 包吃包住排名, 年底双薪排名, 周末双休排名, 加班补助排名, 其他补助排名
]


# 雷达图的指标
c_schema = [
    {"name": "五险一金", "max": 4, "min": 0},
    {"name": "包吃包住", "max": 4, "min": 0},
    {"name": "年底双薪", "max": 4, "min": 0},
    {"name": "周末双休", "max":4, "min": 0},
    {"name": "加班补助", "max": 4, "min": 0},
    {"name": "其他补助", "max": 4, "min": 0},
]

# 创建雷达图对象
c = (
    Radar()
    # 添加雷达图的指标和形状
    .add_schema(schema=c_schema, shape="circle")
    # 添加北京的数据，设置颜色为橙色
    .add("北京", value_bj, color="#f9713c")
    # 添加上海的数据，设置颜色为浅绿色
    .add("上海", value_sh, color="#b3e4a1")
    # 添加广州的数据，设置颜色为蓝色
    .add("广州", value_gz, color="#5c96f2")
    # 添加深圳的数据，设置颜色为紫色
    .add("深圳", value_sz, color="hotpink")
    # 设置系列选项，隐藏标签
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    # 设置全局选项，添加标题
    .set_global_opts(title_opts=opts.TitleOpts(title="北上广深福利排名雷达图"))
    # 渲染为HTML文件
    .render("radar_air_quality.html")
)