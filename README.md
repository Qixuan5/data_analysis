**本项目由上海交通大学安泰经济与管理学院徐海峰副教授指导完成**

杨绮譞：完成房价分析部分

路瑶：完成岗位分析部分

王翌阳：完成交通分析部分

刘婧祺：完成商品分析部分

# 城市多维度发展分析：基于房价、岗位、交通与商品的跨域数据研究

## 📌 项目简介

本项目围绕中国一线城市——北京、上海、广州、深圳，从房价、岗位、交通、商品四个维度出发，结合大规模爬虫采集与可视化分析，构建一个多层次、多角度的城市发展画像。项目内容包括数据采集脚本、数据清洗与处理逻辑、可视化代码及样例分析结果，适用于城市比较、区域特征识别与跨领域研究。


## 📁 项目结构

```
├── README.md                # 本文件
├── 房价分析/
│   ├── data/                # 原始与清洗后房源数据
│   ├── crawl_lianjia.py     # 爬虫脚本（链家）
│   ├── clean_lianjia.py     # 数据清洗与地理坐标获取
│   └── visualize_lianjia.py # 可视化分析
├── 岗位分析/
│   ├── data/
│   ├── crawl_jobs.py        # 58同城招聘信息爬虫
│   ├── clean_jobs.py        # 薪资字段清洗、福利缺失处理
│   └── visualize_jobs.py    # 热力图、词云、桑基图等可视化
├── 交通分析/
│   ├── data/
│   ├── crawl_subway.py      # 爬取高德地铁图数据
│   ├── visualize_subway.py  # 绘制地铁站分布图与词云
│   └── traffic_eval.md      # 拥堵指标分析方法概述
├── 商品分析/
│   ├── data/
│   ├── jd_comment_spider.py # 京东评论爬虫（基于DrissionPage）
│   ├── clean_comment.py     # 评论数据处理与RFM建模
│   └── visualize_comment.py # 消费偏好可视化
```


## 🧾 模块详解

### 1. 房价分析（链家网二手房）

* 📌 目标：抓取北上广深二手房数据，用于结构建模与价格分析。
* 🔍 数据字段：总价、单价、面积、房型、楼层、挂牌时间、交易权属等共43项。
* 🛠 技术栈：

  * requests + BeautifulSoup + 正则表达式
  * CSV写入 + pandas清洗 + AMap API 经纬度解析
* 📊 可视化方法：

  * 分布图、箱线图、Pairplot、KMeans聚类、热力地图、3D可视化
* 📈 核心洞察：

  * 多变量对房价的影响、城市间价格结构差异、豪宅与刚需市场分布等

### 2. 岗位分析（58同城招聘信息）

* 📌 目标：爬取一线城市职位数据，分析岗位结构与福利偏好。
* 🔍 数据字段：职位名称、薪资区间、十项福利标识、地区信息等
* 🛠 技术栈：

  * parsel + requests + CSV
  * 异常值处理 + 正则化薪资提取 + 均值化区间
* 📊 可视化方法：

  * 地图热力图、职业词云、箱型图、3D柱图、桑基图、福利热力矩阵
* 📈 核心洞察：

  * 各区招聘密度与薪资水平差异、热门职业与福利覆盖率分析、岗位聚类

### 3. 交通分析（地铁与拥堵指数）

* 📌 目标：基于高德地图提取地铁站点与线路信息，评估城市通勤便利度。
* 🔍 数据字段：地铁线路、站点名、经纬度、地铁数量/密度/覆盖范围
* 🛠 技术栈：

  * requests + BeautifulSoup + JSON解析
  * Cartopy / Matplotlib / Seaborn / Pyecharts
* 📊 可视化方法：

  * 地铁线路图、站点数量柱状图、词云、线路密度图
* 📈 核心洞察：

  * 上海/北京地铁网完善程度优于深圳、广州；覆盖密度与换乘便利性对比

### 4. 商品分析（京东评论）

* 📌 目标：从京东评论中分析各城市消费者偏好与消费能力。
* 🔍 数据字段：SKU、价格、评分、评论关键词、用户省份、是否图片评论等
* 🛠 技术栈：

  * DrissionPage（自动化浏览器） + 正则抽取 Ajax 请求数据
  * RFM 模型 + 主成分分析 PCA + 城市级关键词聚类
* 📊 可视化方法：

  * 情感热力图、消费关键词词云、雷达图、RFM 分层图、箱线图
* 📈 核心洞察：

  * 一线城市偏好差异显著，关键词体现地域消费文化；RFM划分核心城市

## 🧰 使用说明

1. **环境依赖安装**

   ```bash
   pip install -r requirements.txt
   ```

   或手动安装：

   ```
   pandas, matplotlib, seaborn, requests, beautifulsoup4, drissionpage, pyecharts, scikit-learn, numpy, cartopy
   ```

2. **运行方式**

   * 运行房价爬虫：

     ```bash
     python crawl_lianjia.py
     ```
   * 清洗并获取经纬度：

     ```bash
     python clean_lianjia.py
     ```
   * 各子项目均含 `visualize_*.py` 脚本，可独立运行生成图表。

---

## 🌐 数据来源

* 链家二手房：https\://{city}.lianjia.com/ershoufang/
* 58同城招聘：https\://{city}.58.com/quanzhizhaopin/
* 高德地图地铁数据：[http://map.amap.com/subway/](http://map.amap.com/subway/)
* 京东评论：[https://search.jd.com](https://search.jd.com)


## 💡 项目亮点

* 全流程自动化爬虫 + 多源数据整合
* 四维度综合对比城市画像：房价 × 岗位 × 交通 × 商品
* 高密度可视化图谱，适用于展示、竞赛、研究、政策建议
* 使用 DrissionPage 提高 Ajax 抓取精准性
* 包含从“单指标”到“多指标耦合”的分析方案

