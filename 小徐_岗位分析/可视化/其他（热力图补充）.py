import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.rc("font",family='YouYuan')


# ✅ 读取数据
df = pd.read_csv(r"D:\360MoveData\Users\30648\Desktop\58同城数据\北京\北京1.csv")

# ✅ 福利项字段
welfare_cols = ['五险一金', '包住', '包吃', '年底双薪', '周末双休', '交通补助', '加班补助', '饭补', '话补', '房补']

# ✅ 二值化福利项
welfare_df = df[welfare_cols].fillna("无").applymap(lambda x: 0 if x == "无" else 1)

# ✅ 福利之间 Pearson 相关性矩阵
corr_matrix = welfare_df.corr()

# ✅ 职位类别与福利项的频率统计
category_welfare = df.groupby('类别')[welfare_cols].apply(
    lambda x: x.fillna("无").applymap(lambda v: 0 if v == "无" else 1).mean()
)

# ✅ 选择最多的前5类职位
top_categories = df['类别'].value_counts().nlargest(5).index.tolist()
selected_cats = category_welfare.loc[top_categories]

# 将右侧“原点”节点进一步向左移动，使文字不与连线重叠

fig, ax = plt.subplots(figsize=(13, 13))
ax.set_xlim(-1, len(welfare_cols) + 3.2)
ax.set_ylim(-2, len(welfare_cols) + 1.5)
ax.axis("off")

for i in range(len(welfare_cols)):
    for j in range(i):
        r = corr_matrix.iloc[i, j]
        color = [
        '#F0FFF0',  # 非常浅的绿色（接近白色）
        '#E0F7FA',  # 柔和的浅绿色
        '#D1F2EB',  # 浅绿色
        '#B2DFDB',  # 柔和的绿色
        '#80CBC4',  # 深一点的柔和绿
        '#4DB6AC',  # 柔和的深绿
        '#A5D6A7'   # 淡绿色（柔和的绿）
    ]
            # 根据相关系数的大小选择颜色
        if abs(r) < 0.1:
            selected_color = color[0]  # 非常浅的粉色
        elif abs(r) < 0.15:
            selected_color = color[1]  # 浅粉色
        elif abs(r) < 0.2:
            selected_color = color[2]  
        elif abs(r) < 0.25:
            selected_color = color[3]  # 浅粉色
        elif abs(r) < 0.3:
            selected_color = color[4] # 柔和的粉色
        else:
            selected_color = color[5]  # 深一点的粉色
        rect = patches.FancyBboxPatch(
            (j, len(welfare_cols)-i-1), 1, 1,
            boxstyle="round,pad=0.01", linewidth=0.5,
            facecolor=selected_color, edgecolor='white'
        )
        ax.add_patch(rect)
        ax.text(j + 0.5, len(welfare_cols)-i-0.5, f"{r:.2f}", ha='center', va='center', fontsize=9)
# ✅ 上三角连线（职位类别）
positions = {w: (i, len(welfare_cols) - i - 1) for i, w in enumerate(welfare_cols)}
step = len(welfare_cols) / (len(selected_cats) + 1)
x_dot_offset = len(welfare_cols) + 0.8     # 原点位置
x_text_offset = x_dot_offset + 0.6         # 文本稍微右移，防止重叠

for idx, cat in enumerate(selected_cats.index):
    y = len(welfare_cols) - (idx + 1) * step

    # 原点左移
    ax.plot(x_dot_offset, y, 'o', color='purple', markersize=8)
    ax.text(x_text_offset, y, cat, ha='left', va='center', fontsize=10, fontweight='bold')

    for wid, score in selected_cats.loc[cat].items():
        if score > 0.2:
            wx, wy = positions[wid]
            ax.plot([x_dot_offset, wx + 0.5], [y, wy + 0.5], color='#4DB6AC', linewidth=1.5, alpha=0.7)
            ax.plot(wx + 0.5, wy + 0.5, 'o', color='darkgreen', markersize=4)

# ✅ 福利项标签
for i, name in enumerate(welfare_cols):
    ax.text(-0.5, len(welfare_cols) - i - 0.5, name, ha='right', va='center', fontsize=9)
    ax.text(i + 0.5, -0.8, name, rotation=45, ha='right', va='top', fontsize=9)


plt.tight_layout()
plt.show()



