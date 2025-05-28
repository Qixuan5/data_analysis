import argparse, pathlib, warnings
import pandas as pd, numpy as np,contextily as cx
import matplotlib.pyplot as plt, seaborn as sns
import plotly.express as px, plotly.graph_objects as go
import geopandas as gpd
import pydeck as pdk
import re, jieba
import visualization as vz
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.mixture import GaussianMixture
from gensim import corpora, models
from snownlp import SnowNLP
from pathlib import Path
import networkx as nx
from wordcloud import WordCloud, STOPWORDS
from matplotlib.cm import ScalarMappable
from matplotlib.patches import RegularPolygon
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib.patches import Ellipse
warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False     # 负号正常
# ------------------------------------------------------------------

def load_data(xlsx="总评论final.xlsx") -> pd.DataFrame:
    """读入评论数据并做最基本字段统一"""
    if not pathlib.Path(xlsx).exists():
        raise FileNotFoundError(f"未找到 {xlsx}")
    df = pd.read_excel(xlsx)

    # 把时间列标准化成 datetime
    time_cols = [c for c in df.columns if "时间" in c]
    if time_cols:
        df["评论时间"] = pd.to_datetime(df[time_cols[0]])
    else:
        df["评论时间"] = pd.NaT

    # 确保关键列存在
    for col in ["评论IP", "价格"]:
        if col not in df.columns:
            df[col] = np.nan
    return df


# ------------------------------------------------------------------
# 2. 暗线 1️⃣：招聘热力 → 人口流向
# ------------------------------------------------------------------
def heatmap_city_activity(df: pd.DataFrame,
                          geojson="china_provinces.geojson"):
    """省级评论量热力地图 (PNG)"""
    if not pathlib.Path(geojson).exists():
        print(f"[Skip] 缺 geojson：{geojson}")
        return
    cnt = df["评论IP"].value_counts().reset_index()
    cnt.columns = ["province", "cnt"]
    gdf = gpd.read_file(geojson).merge(
        cnt, left_on="name", right_on="province", how="left").fillna(0)
    vmax = gdf['cnt'].quantile(0.9)          # 去掉最头部 10% 的极端
    gdf['cnt_log'] = np.log1p(np.clip(gdf['cnt'], None, vmax))
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf.plot(column="cnt", cmap="Reds", linewidth=.8,
             ax=ax, edgecolor="0.8", legend=True)
    ax.set_title("Comment volume heat map")
    plt.tight_layout()
    plt.savefig("heatmap_city_activity.png", dpi=300)
    plt.close()
    print("[Saved] heatmap_city_activity.png")

# -------------------------------------------------------------------

def analyze_comment_drivers(df: pd.DataFrame,
                            stats_csv: str = "province_stats.csv",
                            output_csv: str = "comment_driver_metrics.csv",
                            plot: bool = True):
    """
    计算每省评论密度 (条/万人) 并与 GDP、互联网普及率做相关性分析。

    Parameters
    ----------
    df : pd.DataFrame
        原始评论数据，需包含列 “评论IP”（省份名称）。
    stats_csv : str
        含有人口、GDP、互联网普及率等指标的省级 CSV。应至少包含字段:
        province, population, gdp, internet_rate
    output_csv : str
        结果表输出的文件名。
    plot : bool
        是否绘制散点图。
    """
    if not pathlib.Path(stats_csv).exists():
        raise FileNotFoundError(
            f"缺省级统计文件 {stats_csv}，请准备包含 province, population, gdp, internet_rate 的 CSV")
    # 读取统计指标
    stats_df = pd.read_csv(stats_csv)
    stats_cols = {"province", "population", "gdp", "internet_rate"}
    if not stats_cols.issubset(stats_df.columns):
        raise KeyError(f"{stats_csv} 必须至少包含列: {stats_cols}")
    # 统一名称
    def _norm_name(x):
        if pd.isna(x):
            return "未知"

        x = str(x).strip()   
        for suf in ["省", "市", "自治区", "壮族自治区", "回族自治区", "维吾尔自治区"]:
            x = x.replace(suf, "")
        return x.strip()
    stats_df["province"] = stats_df["province"].apply(_norm_name)

    # 评论数 by province
    cnt = (df.loc[df["评论IP"] != "未知", "评论IP"]
             .apply(_norm_name)
             .value_counts()
             .reset_index()
             .rename(columns={"index": "province", "评论IP": "comment_cnt"}))
    cnt.columns=["province", "comment_cnt"]
    merged = stats_df.merge(cnt, on="province", how="left").fillna(0)
    merged["comment_density_per10k"] = merged["comment_cnt"] / (merged["population"] / 10000.0)

    # 相关性
    pearson_gdp = merged[["comment_cnt", "gdp"]].corr(method="pearson").iloc[0, 1]
    spearman_gdp = merged[["comment_cnt", "gdp"]].corr(method="spearman").iloc[0, 1]

    pearson_net = merged[["comment_cnt", "internet_rate"]].corr(method="pearson").iloc[0, 1]
    spearman_net = merged[["comment_cnt", "internet_rate"]].corr(method="spearman").iloc[0, 1]

    print(f"[Correlation] 评论量 vs GDP  Pearson={pearson_gdp:.3f}, Spearman={spearman_gdp:.3f}")
    print(f"[Correlation] 评论量 vs 互联网普及率 Pearson={pearson_net:.3f}, Spearman={spearman_net:.3f}")

    # 导出
    merged.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[Saved] {output_csv}")

    if plot:
        # GDP vs comments scatter
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(merged["gdp"], merged["comment_cnt"])
        ax.set_xlabel("GDP (亿元)")
        ax.set_ylabel("评论量")
        ax.set_title("GDP vs 评论量")
        plt.tight_layout()
        plt.savefig("gdp_vs_comments.png", dpi=300)
        plt.close()
        # Internet vs comments scatter
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(merged["internet_rate"], merged["comment_cnt"])
        ax.set_xlabel("互联网普及率 (%)")
        ax.set_ylabel("评论量")
        ax.set_title("互联网普及率 vs 评论量")
        plt.tight_layout()
        plt.savefig("internet_vs_comments.png", dpi=300)
        plt.close()
        print("[Saved] gdp_vs_comments.png, internet_vs_comments.png")
# -------------------------------------------------------------------
# -------------------------------------------------------------------
#  省份情感指标 + 3D Spike Map
# -------------------------------------------------------------------

# ---------- 1) 计算正面比例 & 平均情感分 --------------------------------
def sentiment_metrics_by_province(df: pd.DataFrame,
                                  geojson_path: str = "china_provinces.geojson",
                                  out_csv: str = "province_sentiment_metrics.csv",
                                  positive_th: float = 0.8) -> pd.DataFrame:
    """
    df 需包含:
        - 评论文本列:  '评论内容'
        - 省份列:      '评论IP'
    输出:
        province, avg_sentiment, positive_ratio, comment_cnt
    """
    def _sent_score(t):
        try:
            return SnowNLP(str(t)).sentiments
        except Exception:
            return 0.5   # 无法解析 → 中性

    df["sentiment"] = df["评论内容"].apply(_sent_score)
    grp = df.groupby("评论IP")
    metrics = grp["sentiment"].agg(
        avg_sentiment="mean",
        comment_cnt="size",
        positive_ratio=lambda s: (s > positive_th).mean()
    ).reset_index().rename(columns={"评论IP": "province"})

    metrics.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[Saved] {out_csv}")

    if geojson_path:  # 可选检查 GeoJSON 省份匹配
        gdf = gpd.read_file(geojson_path)[["name"]]
        missing = set(metrics["province"]) - set(gdf["name"])
        if missing:
            print("[Warning] GeoJSON 缺少省份:", "、".join(missing))
    return metrics



from matplotlib.cm import ScalarMappable

def heatmap_sentiment_spikes(metric_csv: str = "province_sentiment_metrics.csv",
                             geojson: str = "china_provinces.geojson",
                             score_col: str = "avg_sentiment",
                             out_png: str = "heatmap_sentiment_spike.png",
                             spike_scale: float = 2.5):
    """
    参数
    ----
    metric_csv : CSV，至少含 [province, avg_sentiment]
    geojson    : 省级边界，字段 'name'
    score_col  : 使用哪一列作为情感得分（0-1）
    spike_scale: 柱高 *纬度度数*
    """

    # --- 读指标 ---
    mdf = pd.read_csv(metric_csv)[["province", score_col]]
    gdf = (gpd.read_file(geojson)
           .merge(mdf, left_on="name", right_on="province", how="left")
           .fillna(0))

    # --- 底图按情感分上色 ---
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf.plot(column=score_col, cmap="RdYlGn",  # 绿=高分，红=低分
             linewidth=.5, edgecolor="gray",
             legend=False, ax=ax,
             vmin=0.73, vmax=0.86)

    # colorbar
    sm = ScalarMappable(cmap="RdYlGn",
                        norm=plt.Normalize(vmin=0.73, vmax=0.86))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("平均情感分 (0–1)")

    # --- 叠柱 ---
    for _, row in gdf.iterrows():
        if row[score_col]!=0:
            score = row[score_col]-0.7
        if score == 0:
            continue
        lon, lat = row.geometry.centroid.x, row.geometry.centroid.y
        dh       = score * score * spike_scale   *100    # 高度∝平均分
        ax.vlines(lon, lat, lat + dh,
                  colors="lightcyan", linewidth=5, alpha=0.85)
        ax.plot(lon, lat + dh, marker="o",
                color="deepskyblue", markersize=4)

    ax.set_title("省级平均情感分热力图（含柱形标注）")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[Saved] {out_png}")



# ---------- 生成同心环坐标 ----------
def ring_coords(k):
    """返回环 k (半径 k) 的 6*k 坐标，起点东南方向顺时针"""
    if k == 0:
        return [(0, 0)]
    coords = []
    q, r = k, 0  # 起点 (k,0)
    directions = [( -1,  1), (-1, 0), (0, -1),
                  (1, -1),  (1, 0),  (0, 1)]
    for dx, dy in directions:
        for _ in range(k):
            coords.append((q, r))
            q += dx
            r += dy
    return coords

def concentric_coords(n):
    """返回 n 个坐标，按同心环由内到外"""
    coords = []
    k = 0
    while len(coords) < n:
        ring = ring_coords(k)
        coords.extend(ring)
        k += 1
    return coords[:n]

# ---------- 主绘图函数 ----------
def member_hex_spiral(df: pd.DataFrame,
                      value_col: str = "member_pct",
                      out_png: str = "member_hex_spiral.png"):
    """值高排中心，同心蜂窝；标签=省名+百分比"""

    df_sorted = df.sort_values(value_col, ascending=False).reset_index(drop=True)
    coords = concentric_coords(len(df_sorted))
    df_sorted[["q", "r"]] = pd.DataFrame(coords, index=df_sorted.index)

    # 轴向→笛卡尔
    df_sorted["x"] = df_sorted["q"] + df_sorted["r"] / 2
    df_sorted["y"] = np.sqrt(3) / 2 * df_sorted["r"]

    # 颜色
    vmin, vmax = df_sorted[value_col].min(), df_sorted[value_col].max()
    cmap = sns.color_palette("YlGnBu", as_cmap=True)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8, 8))
    # --- 坐标映射 (pointy-top 公式) ---
    radius = 1
    df_sorted["x"] = np.sqrt(3) * (df_sorted["q"] + df_sorted["r"] / 2) * radius
    df_sorted["y"] = 1.5 * df_sorted["r"] * radius

    # --- 绘制 ---
    for _, row in df_sorted.iterrows():
        color = cmap(norm(row[value_col]))
        ax.add_patch(
            RegularPolygon(
                (row.x, row.y),
                numVertices=6,
                radius=radius * 0.97,   # 0.97 避免重叠描边
                orientation=0,          # 尖顶
                facecolor=color,
                edgecolor="white",
                lw=1,
                zorder=2,
            )
        )
        # 省名 + 占比
        ax.text(row.x, row.y + 0.35 * radius, row.province,
                ha="center", va="center", fontsize=12, color="black")
        ax.text(row.x, row.y - 0.35 * radius, f"{row[value_col]:.0%}",
                ha="center", va="center", fontsize=10, color="black")

    # --- 轴范围 ---
    xs, ys = df_sorted["x"], df_sorted["y"]
    ax.set_xlim(xs.min() - 2 * radius, xs.max() + 2 * radius)
    ax.set_ylim(ys.min() - 2 * radius, ys.max() + 2 * radius)


    ax.set_aspect("equal"); ax.axis("off")

    # 色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm._A=[]
    cbar = fig.colorbar(sm, ax=ax, fraction=0.033, pad=0.02)
    cbar.set_label("会员评论占比")

    ax.set_title("省级会员渗透蜂窝图（中心 = 最高占比）")
    plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()
    print(f"[Saved] {out_png}")



def scatter_jobs_vs_comments(df: pd.DataFrame,
                             job_csv="job_cnt.csv"):
    """岗位数 vs 评论量气泡图 (HTML)"""
    if not pathlib.Path(job_csv).exists():
        print(f"[Skip] 缺岗位数据：{job_csv}")
        return
    job = pd.read_csv(job_csv)           # province,job_cnt,avg_salary
    comm = df["评论IP"].value_counts().rename(
        "comment_cnt").reset_index().rename(columns={"index": "province"})
    g = comm.merge(job,left_on='评论IP',right_on='province', how='inner')
    print(comm.shape)   # ① 评论侧行数
    print(job.shape)    # ② 岗位侧行数
    print(g.shape)
    fig = px.scatter(g, x="job_cnt", y="comment_cnt",
                     size="avg_salary", color="province",
                     hover_data=["avg_salary"],
                     title="岗位数 vs 评论量")
    fig.write_html("scatter_jobs_vs_comments.html")
    print("[Saved] scatter_jobs_vs_comments.html")
    png_file = "scatter_jobs_vs_comments.png"
    fig.write_image(png_file)
    print(f"[Saved] {png_file}")


def rfm_heatmap(df, snapshot="2025-05-19"):
    """
    省份级 R(=近30天评论量) - F(累计评论量) - M(平均客单价) 热力图
    - R：近 30 天评论条数，越多越新 → 分 3 桶
    - F：历史评论条数          → 分 3 桶
    - M：平均客单价            → 分 3 桶
    """

    # ---------- 1. 省份字段标准化 ----------
    df['province'] = (df['评论IP']
                      .astype(str)
                      .str.strip()
                      .str.replace('省|市', '', regex=True))
    df = df.dropna(subset=['province'])
    mask_str_nan = df['province'].str.lower() == 'nan'
    df = df[~mask_str_nan]

    # ③ 其它占位符：'未知' '空' 'None' …
    df = df[~df['province'].isin(['未知', '空', 'None', '海外'])]
    # ---------- 2. 时间字段标准化 ----------
    df['评论时间'] = pd.to_datetime(df['评论时间'], errors='coerce')
    snapshot_ts   = pd.Timestamp(snapshot)
    window_start  = snapshot_ts - pd.Timedelta(days=30)

    # ---------- 3. 近 30 天评论量 (Recent30) ----------
    recent_mask  = df['评论时间'] >= window_start
    recent_cnt   = (df[recent_mask]
                    .groupby('province')['评论时间']
                    .size()
                    .rename('Recent30'))

    # ---------- 4. 聚合到省份级 R F M ----------
    rfm = (df.groupby('province')
             .agg(Frequency = ('province', 'size'),
                  Monetary  = ('价格', 'mean'))
             .merge(recent_cnt, left_index=True, right_index=True, how='left')
             .fillna({'Recent30': 0}))      # 无近 30 天评论 → 0

    # ---------- 5. 分箱：等频 3 桶 ----------
    rfm['R'] = pd.qcut(rfm['Recent30'], 3, labels=False, duplicates='drop') + 1
    rfm['F'] = pd.qcut(rfm['Frequency'],3, labels=False, duplicates='drop') + 1
    rfm['M'] = pd.qcut(rfm['Monetary'], 3, labels=False, duplicates='drop') + 1

    # ---------- 6. 透视 & 绘图 ----------
    pivot = (rfm.pivot_table(values='Monetary',
                             index='F', columns='R',
                             aggfunc='mean')
                  .fillna(0))

    plt.figure(figsize=(6,4))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap='YlOrRd')
    plt.title("省份级 R(近30天评论量)-F-M 价值热力图 (数字=平均客单价¥)")
    plt.tight_layout()
    plt.savefig("province_rfm_heatmap.png", dpi=300)
    plt.close()
    print("[Saved] province_rfm_heatmap.png")
    rfm_reset = rfm.reset_index()      # 把索引 province 变成列

    province_grid = (rfm_reset
                    .groupby(['F','R'])['province']
                    .apply(lambda x: ", ".join(x))
                    .unstack(fill_value="")          # 行=F，列=R
    )

    print(province_grid)
    province_grid.to_excel("RF_province_list.xlsx")


# ------------------------------------------------------------------

def boxplot_price_tier(df: pd.DataFrame):
    """
    把评论按省份分成 4 组：
        • 北京
        • 上海
        • 广东
        • Other（其余省份/城市）
    并画客单价箱线图
    """
    # -------- 1. 先提取省份 --------
    df['province'] = (df['评论IP']
                      .astype(str)
                      .str.strip()
                      .str.replace('省|市', '', regex=True))

    # -------- 2. 建映射字典 --------
    mapping = {
        '北京' : 'Beijing',
        '上海' : 'Shanghai',
        '广东' : 'Guangdong'
    }
    df['tier'] = df['province'].map(mapping).fillna('Other')

    # -------- 3. 画箱线图 --------
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='tier', y='价格', order=['Beijing','Shanghai','Guangdong','Other'])
    plt.title('客单价分布：Beijing / Shanghai / Guangdong / Other')
    plt.ylabel('价格 (¥)')
    plt.xlabel('Province Tier')
    plt.tight_layout()
    plt.savefig('box_price_tier_4groups.png', dpi=300)
    plt.close()
    print('[Saved] box_price_tier_4groups.png')


def gmm_customer_segments(df: pd.DataFrame,
        meta_file: str = "job_cnt_with_house.xlsx",
        output: str = "bjg_other_radar_pretty.html",
        SCALE=100) -> None:
    """
    北京 / 上海 / 广东 / Other 三维画像雷达图
    轴: 房价(元/㎡)  工资(元/月)  客单价(元)  (归一化 0-1)

    参数
    ----
    df         : 评论级 DataFrame，须含 列 评论IP, 价格
    meta_file  : 含省级 avg_salary, avg_house_price 的 Excel
    output     : 生成的 HTML 文件名
    """
    # ---------------- 1. 读省级 meta ----------------
    meta = (pd.read_excel(meta_file)
              .rename(columns={'avg_salary':      '工资',
                               'house_price': '房价',
                               'province':        'province'}))
    meta['province'] = meta['province'].str.strip()
    meta['工资'] = meta['工资'] / 4 
    # -------- 2. 评论省份 & 合并 ----------
    df['province'] = (df['评论IP'].astype(str)
                      .str.strip()
                      .str.replace('省|市','', regex=True))
    df = df.merge(meta, on='province', how='left').dropna(subset=['房价','工资'])

    # -------- 3. 四组映射 ----------
    gp_map = {'北京':'北京','上海':'上海',
              '广东':'广东','广州':'广东','深圳':'广东'}
    df['group'] = df['province'].map(gp_map).fillna('Other')

    # -------- 4. 计算均值 & 放大客单价 ----------
    feat = ['房价','工资','价格']
    grp = df.groupby('group')[feat].mean()
    grp['客单价×' + str(SCALE)] = grp['价格'] * SCALE    # 新列
    grp = grp.drop(columns='价格')

    # -------- 5. 画雷达 ----------
    color = {'北京':'#E24A33','上海':'#348ABD',
             '广东':'#988ED5','Other':'#2CA02C'}
    theta = ['房价(元/㎡)','工资(元/季度)',f'客单价×{SCALE}(元)']
    order = ['Other','北京','上海','广东']

    fig = go.Figure()
    for g in order:
        if g not in grp.index:
            continue
        r = grp.loc[g]
        fig.add_trace(go.Scatterpolar(
            r=r.values,
            theta=theta,
            fill='toself' if g!='Other' else 'none',
            line=dict(color=color[g],
                      width=3 if g!='Other' else 4,
                      dash='solid' if g!='Other' else 'dot'),
            marker=dict(size=6),
            name=g,
            hovertemplate="<b>%{fullData.name}</b><br>" +
                          "%{theta}: %{r:,.0f}<extra></extra>"
        ))

    fig.update_layout(
        title=dict(text=f"北京 / 上海 / 广东 / Other ─ (客单价 × {SCALE})", x=0.5),
        polar=dict(radialaxis=dict(showline=True, gridcolor="#CCCCCC")),
        legend=dict(orientation='h', y=-0.1)
    )
    fig.write_html(output)
    print(f"[Saved] {output}")
    fig.write_image("bjg_other_radar_pretty.png")
    print(f"[Saved] {"bjg_other_radar_pretty.png"}")


# ------------------------------------------------------------------
# 5. 暗线 4️⃣：城市性格
# ------------------------------------------------------------------
def city_wordcloud(df, city="北京"):
    txt = " ".join(df.loc[df['评论IP'] == city, '评论内容']
                     .dropna().astype(str))
    if not txt:
        print(f"[Skip] {city} 无评论")
        return

    # ---- 自动找字体 ----
    def pick_font():
        # 1) 本目录 simhei.ttf
        if Path("simhei.ttf").exists():
            return "simhei.ttf"
        # 2) macOS 系统 PingFang
        pf = Path("/System/Library/Fonts/PingFang.ttc")
        if pf.exists():
            return str(pf)
        # 3) Windows 微软雅黑
        win = Path("C:/Windows/Fonts/msyh.ttc")
        if win.exists():
            return str(win)
        raise FileNotFoundError("找不到可用中文字体，请安装 SimHei 或指定 font_path")

    font = pick_font()

    wc = WordCloud(
        width=800, height=600,
        font_path=font,
        background_color="white",
        stopwords=set(STOPWORDS)
    ).generate(txt)

    out = f"wordcloud_{city}.png"
    wc.to_file(out)
    print(f"[Saved] {out}")


STOP_PATH = "stopwords_zh.txt"   # ① 下载好的中文停用词文件

def lda_sankey(df: pd.DataFrame, city="上海", n_topics=5, topn=8):
    if "评论内容" not in df.columns:
        print("[Skip] 缺 评论内容")
        return

    docs_raw = (df.loc[df["评论IP"] == city, "评论内容"]
                  .dropna().astype(str).tolist())
    if not docs_raw:
        print(f"[Skip] {city} 无文本")
        return

    # ---------- 1. 预处理 ----------
    stop = set([l.strip() for l in open(STOP_PATH, encoding='utf-8')])
    texts = []
    for text in docs_raw:
        text = re.sub(r"[^\u4e00-\u9fa5]+", " ", text)   # 去标点
        seg   = jieba.lcut(text)
        seg_f = [w for w in seg if len(w) > 1 and w not in stop]
        if seg_f:
            texts.append(seg_f)

    if not texts:
        print(f"[Skip] {city} 过滤后无有效文本")
        return

    # ---------- 2. LDA ----------
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(t) for t in texts]
    lda = models.LdaModel(corpus, num_topics=n_topics,
                          id2word=dictionary,
                          passes=10, random_state=0)

    # ---------- 3. 构造 Sankey 数据 ----------
    topics = [f"Topic {i}" for i in range(n_topics)]
    words, weights = [], []
    for i in range(n_topics):
        for w, p in lda.show_topic(i, topn=topn):
            words.append(w)
            weights.append(p)

    src, dst, val = [], [], []
    for i in range(n_topics):
        for j in range(topn):
            src.append(i)
            dst.append(n_topics + i*topn + j)
            val.append(weights[i*topn + j])

    labels = topics + words
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=labels, pad=15, thickness=15),
        link=dict(source=src, target=dst, value=val)
    ))
    fig.update_layout(title=f"LDA Topic Sankey – {city}",
                      font=dict(size=12))
    out = f"lda_sankey_{city}.html"
    fig.write_html(out)
    print(f"[Saved] {out}")

def city_fashion_network(
        df: pd.DataFrame,
        ip_col="评论IP",          # 城市 / IP 字段
        keyword_col="关键词",      # 服装关键词字段
        kw_threshold=1500,        # 关键词出现阈值
        edge_threshold=30,        # 画边的最小评论量
        layout="spring",          # 布局: spring/bipartite/kamada_kawai/radial_bipartite
        figsize=(20, 20),         # 画布尺寸
        save_path="city_clothing_network.png", # 保存路径，None则只显示
        dpi=300,                  # 分辨率
        font_family="sans-serif", # 中文字体
        seed=42,                  # 随机种子
        # 边样式参数
        edge_color_map='Blues',  # 边的颜色映射
        cmap_start=0.2,          # 颜色映射起点 (0到1)
        cmap_end=0.5,            # 颜色映射终点 (0到1)
        edge_width_factor=0.08,  # 边宽度缩放因子
        edge_alpha=0.5,          # 边透明度
        # 节点样式参数
        city_color='#446BB3',    # 城市节点颜色 (Matplotlib Red)
        keyword_color='#1f77b4', # 关键词节点颜色 (Matplotlib Blue)
        node_alpha=0.9,          # 节点透明度
        # 标签样式参数
        min_font_size=12,         # 最小字体大小
        max_font_size=28,        # 最大字体大小
        label_offset=0.028,
        min_node_size=200,       # 最小视觉节点大小
        max_node_size=2500,         # 标签水平偏移量
    ):
    
    print("开始绘制网络图...")

    # 1. 数据准备
    df = df.dropna(subset=[ip_col, keyword_col]) # 确保关键列无缺失值
    keyword_counts = df[keyword_col].value_counts()
    keywords = keyword_counts[keyword_counts >= kw_threshold].index.tolist()
    cities_in_data = df[ip_col].unique().tolist()

    edges_df = (
        df[df[keyword_col].isin(keywords)]
        .groupby([ip_col, keyword_col])
        .size()
        .reset_index(name="weight")
        .query("weight >= @edge_threshold")
    )
    if edges_df.empty:
        print("错误：没有满足条件的边，无法绘制网络图。请检查阈值设置。")
        return

    # 2. 初始化 NetworkX 图 & 计算原始大小
    G = nx.Graph()
    city_sizes_raw = df.groupby(ip_col).size()
    keyword_sizes_raw = df[df[keyword_col].isin(keywords)][keyword_col].value_counts()
    raw_sizes = {} # 存储原始大小

    for city in cities_in_data:
        raw_sizes[city] = city_sizes_raw.get(city, 1)
        G.add_node(city, type="city", raw_size=raw_sizes[city])

    for kw in keywords:
        raw_sizes[kw] = keyword_sizes_raw.get(kw, 1)
        G.add_node(kw, type="keyword", raw_size=raw_sizes[kw])

    for _, r in edges_df.iterrows():
        if G.has_node(r[ip_col]) and G.has_node(r[keyword_col]):
             G.add_edge(r[ip_col], r[keyword_col], weight=r["weight"])

    G.remove_nodes_from(list(nx.isolates(G)))
    if not G.nodes(): print("错误：图中没有节点，无法绘制。"); return

    # 线性缩放节点大小
    valid_raw_sizes_list = [G.nodes[n]['raw_size'] for n in G.nodes()]
    min_raw, max_raw = min(valid_raw_sizes_list), max(valid_raw_sizes_list)

    for n in G.nodes():
        raw_s = G.nodes[n]['raw_size']
        plot_s = min_node_size
        if max_raw > min_raw:
            norm_s = (raw_s - min_raw) / (max_raw - min_raw)
            plot_s = min_node_size + norm_s * (max_node_size - min_node_size)
        G.nodes[n]['size'] = plot_s # 设置最终绘图大小

    # 3. 布局计算
    pos = {}
    city_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "city"]
    keyword_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "keyword"]
    if layout == "spring": pos = nx.spring_layout(G, k=0.8, iterations=50, seed=seed)
    elif layout == "bipartite":
        pos.update({n: (0, i) for i, n in enumerate(city_nodes)})
        pos.update({n: (1, i* (len(city_nodes)/len(keyword_nodes)) if len(keyword_nodes)>0 else 0 ) for i, n in enumerate(keyword_nodes)})
    elif layout == "radial_bipartite":
        num_cities = len(city_nodes); R_city = 0.3
        if num_cities > 0:
            city_angles = np.linspace(0, 2 * np.pi, num_cities, endpoint=False)
            for i, city in enumerate(city_nodes): pos[city] = (R_city * np.cos(city_angles[i]), R_city * np.sin(city_angles[i]))
        num_keywords = len(keyword_nodes); R_kw = 1.0
        if num_keywords > 0:
            kw_angles = np.linspace(0, 2 * np.pi, num_keywords, endpoint=False)
            for i, kw in enumerate(keyword_nodes): pos[kw] = (R_kw * np.cos(kw_angles[i]), R_kw * np.sin(kw_angles[i]))
    else: pos = nx.kamada_kawai_layout(G)
    label_pos = {n: (x + label_offset, y) for n, (x, y) in pos.items()}

    # 4. 绘图
    plt.figure(figsize=figsize)

    # 绘制节点
    node_sizes_dict = {n: G.nodes[n]["size"] for n in G.nodes()}
    plot_node_sizes = list(node_sizes_dict.values())
    plot_node_colors = [city_color if G.nodes[n]["type"] == "city" else keyword_color for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=plot_node_sizes, node_color=plot_node_colors, alpha=node_alpha)

    # 绘制边
    edges = G.edges()
    if edges:
        weights = [G[u][v]["weight"] for u, v in edges]
        plot_edge_widths = [w * edge_width_factor for w in weights]
        norm = mcolors.Normalize(vmin=min(weights), vmax=max(weights))
        original_cmap = plt.get_cmap(edge_color_map)
        start, end = max(0, min(cmap_start, 1)), max(0, min(cmap_end, 1))
        if start >= end: start, end = 0.2, 0.9
        cmap = mcolors.LinearSegmentedColormap.from_list(f"{edge_color_map}_segment", original_cmap(np.linspace(start, end, 256)))
        plot_edge_colors = [cmap(norm(w)) for w in weights]
        nx.draw_networkx_edges(G, pos, width=plot_edge_widths, alpha=edge_alpha, edge_color=plot_edge_colors)

    # 绘制标签
    min_plot_s, max_plot_s = min(plot_node_sizes), max(plot_node_sizes)
    for n, (x, y) in label_pos.items():
        size = node_sizes_dict[n]
        fs = min_font_size
        if max_plot_s > min_plot_s:
            normalized_size = (size - min_plot_s) / (max_plot_s - min_plot_s)
            fs = min_font_size + normalized_size * (max_font_size - min_font_size)
        plt.text(x, y, n, size=fs, ha='left', va='center', fontfamily=font_family, color='black')

    # 5. 美化与保存
    plt.axis("off")
    plt.title(f"省份服装消费结构 ({layout} 布局)", fontfamily=font_family, fontsize=24) # 增大标题字体
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=dpi, bbox_inches="tight"); print(f"✅  已保存到：{save_path}")
    plt.show()
    print("网络图绘制完成。")

def city_level_distribution(
        data_path='总评论final.xlsx',
        ip_col='评论IP',
        top_n=15,
        figsize=(10, 8),
        save_path='city_level_distribution.png',
        dpi=300,
        font_family='sans-serif',  # 默认中文字体，可根据本地环境更换
    ):
    """
    绘制城市等级分布极坐标玫瑰图。

    Parameters:
    ----------
    data_path : str
        数据文件路径。
    ip_col : str
        表中代表城市或地区的字段名。
    top_n : int
        展示的城市数量。
    figsize : tuple
        画布尺寸。
    save_path : str
        图像保存路径。
    dpi : int
        图像分辨率。
    font_family : str
        中文字体。
    """
    # 加载数据
    df = pd.read_excel(data_path)

    # 城市评论数量统计
    city_counts = df[ip_col].value_counts().reset_index()
    city_counts.columns = ['城市', '评论数量']
    city_counts = city_counts[city_counts['城市'] != '未知']

    # 选取前top_n个城市
    top_cities = city_counts.head(top_n)

    # 计算角度
    angles = np.linspace(0, 2 * np.pi, len(top_cities), endpoint=False)

    # 极坐标绘图
    plt.figure(figsize=figsize)
    ax = plt.subplot(111, polar=True)

    bars = ax.bar(angles, top_cities['评论数量'], width=0.4, alpha=0.8,
                  color=plt.cm.Set3(np.linspace(0, 1, len(top_cities))))

    # 设置标签
    ax.set_xticks(angles)
    ax.set_xticklabels(top_cities['城市'], fontfamily=font_family, fontsize=12)
    # 添加数量标签
    for angle, bar, count in zip(angles, bars, top_cities['评论数量']):
        ax.text(angle, bar.get_height() + max(top_cities['评论数量'])*0.05, str(count),
                ha='center', va='bottom', fontsize=12, fontfamily=font_family, color="#92B7FC")
    # 标题
    ax.set_title('城市等级分布图（按评论数量）', fontsize=16, fontfamily=font_family, pad=20)

    plt.tight_layout()

    # 保存图片
    plt.savefig(save_path, dpi=dpi)
    print(f'✅ 图像已保存到 {save_path}')

    plt.show()

from adjustText import adjust_text

def city_keyword_pca_cluster(
        data_path='总评论final.xlsx',
        ip_col='评论IP',
        keyword_col='关键词',
        n_clusters=2,
        figsize=(12, 10),
        save_path='city_keyword_pca_cluster.png',
        dpi=300,
        font_family='sans-serif',  # 根据本地字体环境调整
    ):
    """
    绘制城市-关键词PCA聚类散点图，并标注城市名称（自动调整位置避免重叠）。

    Parameters:
    ----------
    data_path : str
        数据文件路径。
    ip_col : str
        城市或IP字段名。
    keyword_col : str
        关键词字段名。
    n_clusters : int
        聚类的数量。
    figsize : tuple
        画布尺寸。
    save_path : str
        图像保存路径。
    dpi : int
        图像分辨率。
    font_family : str
        中文字体。
    """
    # 数据读取
    df = pd.read_excel(data_path)

    # 创建城市-关键词频率矩阵
    city_keyword_matrix = pd.crosstab(df[ip_col], df[keyword_col])

    # PCA降维
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(city_keyword_matrix)

    # KMeans聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(reduced_data)

    plt.figure(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    texts = []  # 用于自动调整位置的文本对象列表

    for i, color in enumerate(colors):
        cluster_points = reduced_data[labels == i]
        city_names = city_keyword_matrix.index[labels == i]

        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                    c=[color], label=f'Cluster {i+1}', alpha=0.6)

        if len(cluster_points) > 1:
            cov = np.cov(cluster_points, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            order = eigenvalues.argsort()[::-1]
            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
            vx, vy = eigenvectors[:, 0]
            theta = np.degrees(np.arctan2(vy, vx))
            width, height = 2 * np.sqrt(eigenvalues) * 2

            ellipse = Ellipse(xy=np.mean(cluster_points, axis=0),
                              width=width, height=height, angle=theta,
                              color=color, alpha=0.2)
            plt.gca().add_patch(ellipse)

        # 标注每个点的城市名称 (先记录到列表中以便自动调整位置)
        for point, city in zip(cluster_points, city_names):
            texts.append(plt.text(point[0], point[1], city, fontsize=8, 
                                  fontfamily=font_family, alpha=0.8))

    # 自动调整文本位置，避免重叠
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='grey', lw=0.5))

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontfamily=font_family)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontfamily=font_family)
    plt.title('PCA 聚类散点图（城市-关键词）', fontsize=16, fontfamily=font_family)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    print(f'✅ 图像已保存到 {save_path}')
    plt.show()
# ------------------------------------------------------------------
# 6. 主入口
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all",
                        choices=["all", "heatmap", "scatter",
                                 "box_commute", "rfm",
                                 "price_tier", "gmm",
                                 "wordcloud", "lda"])
    parser.add_argument("--city", default="北京")
    args = parser.parse_args()

    df = load_data()

    if args.task in ("all", "heatmap"):
        heatmap_city_activity(df)
        analyze_comment_drivers(df)
        #province_sentiment_metrics.csv 必含列: province, avg_sentiment
        heatmap_sentiment_spikes(metric_csv="province_sentiment_metrics.csv",
                         geojson="china_provinces.geojson",
                         score_col="avg_sentiment",
                         spike_scale=3)   # 可调高度

        member_metrics = (df.groupby("评论IP")["是否是会员"]
                    .mean()
                    .reset_index()
                    .rename(columns={"评论IP":"province",
                                     "是否是会员":"member_pct"}))

        member_hex_spiral(member_metrics)

    if args.task in ("all", "scatter"):
        scatter_jobs_vs_comments(df)
    if args.task in ("all", "rfm"):
        rfm_heatmap(df)
    if args.task in ("all", "price_tier"):
        boxplot_price_tier(df)
    if args.task in ("all", "gmm"):
        gmm_customer_segments(df)
    for city in ['北京','上海','广东']:
        if args.task in ("all", "wordcloud"):
            city_wordcloud(df, city)
        if args.task in ("all", "lda"):
            lda_sankey(df, city)
    
    target_cities = ['北京', '上海', '广东','天津','重庆',"山东"]
    df_filtered = df[df['评论IP'].isin(target_cities)].copy()
    vz.city_fashion_network(df_filtered,kw_threshold=100, edge_threshold=10)
    vz.city_level_distribution()
    vz.city_keyword_pca_cluster()
if __name__ == "__main__":
    main()
