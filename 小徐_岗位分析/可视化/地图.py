import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体
matplotlib.rc("font", family='YouYuan')

# 1. 读取数据
file_path = r"D:\360MoveData\Users\30648\Desktop\58同城数据\深圳\深圳1.csv"
df = pd.read_csv(file_path)

# 2. 福利字段
welfare_fields = [
    "五险一金", "包住", "包吃", "年底双薪", "周末双休",
    "交通补助", "加班补助", "饭补", "话补", "房补"
]

# 3. 计算福利数量
df['福利数量'] = df[welfare_fields].apply(lambda row: sum(x != '无' for x in row), axis=1)

# 4. 按地区汇总
district_stats = df.groupby('地区').agg(
    平均福利=('福利数量', 'mean'),
    平均薪资=('平均薪资', 'mean')
).reset_index()

# 确保为数值类型
district_stats['平均福利'] = pd.to_numeric(district_stats['平均福利'], errors='coerce')
district_stats['平均薪资'] = pd.to_numeric(district_stats['平均薪资'], errors='coerce')
district_stats = district_stats.dropna(subset=['平均福利', '平均薪资'])

# 5. 读取地图
map_path = r"C:\Users\30648\Downloads\深圳市\深圳市.shp"
gdf = gpd.read_file(map_path, encoding='utf-8')
gdf = gdf.rename(columns={"name": "地区"})  # 根据你的地图字段名调整

# 6. 合并数据
merged = gdf.set_index('地区').join(district_stats.set_index('地区'))

# 7. 绘图
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# 定义低饱和度的橙色渐变
colors = [
    '#FADBDF',  # 非常浅的粉色（接近白色）
    '#F5B9C2',  # 浅粉色
    '#F6BDC5',  # 柔和的粉色
    '#F3ABB6',  # 深一点的粉色
]

# 创建自定义色卡
cmap_name = 'custom_low_saturation_orange'
n_bins = 100  # 分成10个渐变区间
custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

# ✅ 地图热力图
merged.plot(
    column='平均福利',
    ax=ax,
    legend=True,
    cmap=custom_cmap,
    edgecolor='white',
    linewidth=1,
    missing_kwds={  # ⬅️ 控制无数据区颜色
        'color': 'white',     # 低饱和度橙色
        'edgecolor': 'white',
        'hatch': None,
        'label': '无数据'
    }
)

# ✅ 添加薪资点（使用紫色调 colormap）
scatter = ax.scatter(
    merged.geometry.centroid.x,
    merged.geometry.centroid.y,
    s=merged['平均薪资'] * 0.01,
    c=merged['平均薪资'],
    cmap='Blues',       # 对比色：紫色系
    alpha=0.85,
    linewidth=0.5
)

# ✅ 添加颜色条（薪资）
cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, pad=0.02)
cbar.set_label('平均薪资（元）')
cbar.outline.set_visible(False)

# 图标题与布局
plt.title('深圳各区平均福利数量与平均薪资分布图', fontsize=15)
plt.axis('off')
plt.tight_layout()
plt.show()
