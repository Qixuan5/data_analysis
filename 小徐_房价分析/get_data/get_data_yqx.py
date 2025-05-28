# 发送请求
import requests  # 数据请求模块
from bs4 import BeautifulSoup
import re
import csv
import time
import random

# other 函数
def extract_house_info(house_info):
    # 初始化默认值
    num_rooms = area = orientation = renovation = floor = build_time = house_type = None
    
    for info in house_info:
        info = info.strip()  # 去除前后空格
        if re.match(r'\d室\d厅', info):  # 几室几厅
            num_rooms = info
        elif re.match(r'\d+\.?\d*平米', info):  # 房屋面积
            area = info
        elif re.match(r'[东南西北]+', info):  # 房屋朝向
            orientation = info
        elif info in ['精装', '简装', '毛坯', '其他']:  # 装修
            renovation = info
        elif '楼层' in info or '层' in info:  # 楼层信息
            floor = info
        elif re.match(r'\d{4}年', info):  # 建立时间
            build_time = info
        elif info in ['板楼', '板塔结合', '塔楼']:  # 户型
            house_type = info
    
    return num_rooms, area, orientation, renovation, floor, build_time, house_type


def extract_tag_info(tag_info):
    # 初始化默认值
    vr_info = taxfree_info = subway_info = five_info = haskey_info = None
    
    for tag in tag_info:
        tag_text = tag.text.strip()  # 获取标签文本并去除空白字符
        if 'VR' in tag_text:  # VR相关标签
            vr_info = tag_text
        elif '房本满五年' in tag_text:  # 房本满五年
            taxfree_info = tag_text
        elif '近地铁' in tag_text:  # 近地铁
            subway_info = tag_text
        elif '房本满两年' in tag_text:  # 房本满两年
            five_info = tag_text
        elif '随时看房' in tag_text:  # 随时看房
            haskey_info = tag_text
    
    return vr_info, taxfree_info, subway_info, five_info, haskey_info



# 子模块爬取函数
def crawl_detail_page(url, headers):
    # 预定义所有字段模板
    detail_template = {
        # 新增字段
        '分区': None,
        '小区名称': None,
        '所在区域': None,
        # 原有上模块字段
        '房屋户型': None,
        '建筑面积': None,
        '套内面积': None,
        '房屋朝向': None,
        '装修情况': None,
        '供暖方式': None,
        '楼层高度': None,
        '所在楼层': None,
        '户型结构': None,
        '建筑类型': None,
        '建筑结构': None,
        '梯户比例': None,
        '配备电梯': None,
        # 下模块字段
        '挂牌时间': None,
        '上次交易': None,
        '房屋年限': None,
        '抵押信息': None,
        '交易权属': None,
        '房屋用途': None,
        '产权所属': None,
        '房本备件': None
    }


    try:
        time.sleep(random.uniform(1, 3))
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            return detail_template

        soup = BeautifulSoup(response.text, 'html.parser')
        detail_data = detail_template.copy()  # 使用模板初始化

        # 新增分区提取逻辑
        def extract_district(soup):
            bread_links = soup.select('.intro.clear[mod-id="lj-common-bread"] a[href*="ershoufang"]')
            # 提取逻辑优化
            if len(bread_links) >= 2:
                # 典型结构：北京二手房 > 顺义二手房 > 李桥二手房
                district_text = bread_links[1].text  # 第二个二手房链接
                return district_text.replace('二手房', '')
            else:
                return None

        # 在数据初始化后添加
        detail_data = detail_template.copy()
        detail_data['分区'] = extract_district(soup)  # 新增提取

        # 提取小区名称
        community_elem = soup.select_one('.communityName a.info')
        if community_elem:
            detail_data['小区名称'] = community_elem.text.strip()

        # 提取所在区域（需要处理可能的颜色样式）
        area_elem = soup.select_one('.areaName .info')
        if area_elem:
            # 去除可能存在的颜色样式代码
            area_text = area_elem.text.strip()
            detail_data['所在区域'] = re.sub(r'style=".*?"', '', area_text).strip()


        # 通用数据提取函数
        def extract_section(section_selector, fields):
            section = soup.select_one(section_selector)
            if not section:
                return

            for li in section.select('li'):
                try:
                    label_elem = li.select_one('.label')
                    if not label_elem:
                        continue

                    # 清洗标签文本
                    label = label_elem.text.strip('：').replace(' ', '')
                    
                    # 特殊处理抵押信息
                    if label == '抵押信息':
                        value_elem = li.select_one('span[title]')
                        value = value_elem['title'] if value_elem else li.text.strip()
                    else:
                        # 提取值并清洗
                        raw_value = li.text.replace(label_elem.text, '').strip()
                        value = raw_value.strip('“”').strip()  # 去除中文引号

                    # 统一空值处理
                    value = value if value and value not in ['暂无数据', '暂无'] else None
                    
                    # 更新到对应字段
                    if label in fields:
                        detail_data[label] = value
                except Exception as e:
                    print(f"字段解析异常: {str(e)}")
                    continue

        # 提取上模块（基本属性）
        extract_section('.base .content ul', [
            '房屋户型', '建筑面积', '套内面积', '房屋朝向', '装修情况',
            '供暖方式', '楼层高度', '所在楼层', '户型结构', '建筑类型',
            '建筑结构', '梯户比例', '配备电梯'
        ])

        # 提取下模块（交易属性）
        extract_section('.transaction .content ul', [
            '挂牌时间', '上次交易', '房屋年限', '抵押信息',
            '交易权属', '房屋用途', '产权所属', '房本备件'
        ])

        return detail_data

    except Exception as e:
        print(f"详情页抓取异常: {str(e)}")
        return detail_template



'''
# 登录功能
def login(session):
    login_url = 'https://sh.lianjia.com/login' 
    login_data = {
        'username': '13043253917', 
        'password': 'yqx251406'
    }
    response = session.post(login_url, data=login_data)
    if response.status_code == 200:
        print("登录成功")
        return True
    else:
        print("登录失败")
        return False
'''

# 写入数据
'''
f = open('测试数据.csv', mode='w', encoding='utf-8',newline='') # 修改1
csv_writer = csv.DictWriter(f, fieldnames=['房屋户型', '建筑面积', '套内面积', '房屋朝向', '装修情况','供暖方式', '楼层高度', '所在楼层', '户型结构', '建筑类型','建筑结构', '梯户比例', '配备电梯','挂牌时间', '上次交易', '房屋年限', '抵押信息','交易权属', '房屋用途', '产权所属', '房本备件','标题', '总价', '单价','地区', '几室几厅', '房屋面积', '楼层', '建立时间', '户型', '关注人数', '发布日期', 'VR看装修', '房本满五年', '近地铁', '房本满两年', '随看房','详情页'])
csv_writer.writeheader()
'''

# 修改1
with open('D:/data_analysis/get_data/data/深圳数据(51-100页).csv', mode='w', encoding='utf-8', newline='') as f:
    csv_writer = csv.DictWriter(f, fieldnames=[
        '分区','小区名称', '所在区域', '房屋户型', '建筑面积', '套内面积', '房屋朝向', '装修情况', 
        '供暖方式', '楼层高度', '所在楼层', '户型结构', '建筑类型',
        '建筑结构', '梯户比例', '配备电梯', '挂牌时间', '上次交易',
        '房屋年限', '抵押信息', '交易权属', '房屋用途', '产权所属',
        '房本备件', '标题', '总价', '单价', '地区', '几室几厅',
        '房屋面积', '楼层', '建立时间', '户型', '关注人数', '发布日期',
        'VR看装修', '房本满五年', '近地铁', '房本满两年', '随时看房', '详情页'
    ])
    csv_writer.writeheader()
    
    # 修改2
    for page in range(51,101):
    # 获取数据 获取网页源代码
    # 发送请求的url地址
        time.sleep(10)
        url = f'https://sz.lianjia.com/ershoufang/pg{page}/'   # 修改3
        print(f"正在爬取：第{page}页")
        # print(f"正在处理page{page}")
        # headers: 请求头（伪装成浏览器）

        '''
        session = requests.Session()
        if not login(session):
            print(f"第{page}页无法继续爬取，请检查登录信息")
            continue
        '''

        # 修改4
        cookie = 'SECKEY_ABVK=k4PdJy4EVPj9Kna1oWxRIEHoXorcblzw5Ar2CnGKQoMgRekeZ3XNzMh2tAN3xHbaVtEwD78v76+MO9/xvwBNQw%3D%3D; BMAP_SECKEY=k4PdJy4EVPj9Kna1oWxRIEHoXorcblzw5Ar2CnGKQoNsOVargXryc26d0K0j1L0CNTU57oo58hsIfT3uFew-6xdW6qTfQMEg-B_65GQN0yM1ONGXoF5-wV6OJHOUQlQrSgjJ50ghVzEWMlX-L2tYUDEOavYCTQCuGGDbxbK-4bdBnSfad-dyzJuynHaF1G-FqQRfXQdA7fCG1tp1ip1ozQ; lianjia_uuid=95bfd471-70ba-41db-ac28-3a4b712e0817; _ga=GA1.2.407692691.1745574148; _ga_WGKDF6B591=GS1.2.1745574148.1.0.1745574148.0.0.0; _ga_RCTBRFLNVS=GS1.2.1745580566.1.1.1745580597.0.0.0; lfrc_=f2ba6bd8-3fa8-4539-b2f6-b7353cfdf410; crosSdkDT2019DeviceId=-qll0p--vrgdor-poe33fkw3qoaju9-kj23zmjt8; Hm_lvt_efa595b768cc9dc7d7f9823368e795f1=1745646265; login_ucid=2000000207262243; lianjia_token=2.0013dfed3e752e12f10272c40fca81694e; lianjia_token_secure=2.0013dfed3e752e12f10272c40fca81694e; security_ticket=cA63E9KjOV0NYs/1FcqwQStOi1INNpnlHyT+LfbQtmQKHuJH8zORbpwKHi6ZNNJo3TzSqWpoxE8Enprlw7WaOwxCkRJ7ZEBdixBWkIYCtFYHx15NYcgOvLSRaR6/VTuDMnnbgAwvu6NwDpxDCVnXvaVi5DB2iZM2u1MHrEt3YzU=; ftkrc_=c07c52df-2d82-4c51-a83d-9736bac7c8e6; _gid=GA1.2.474775769.1745742253; _ga_1W6P4PWXJV=GS1.2.1745757401.1.1.1745757406.0.0.0; _ga_W9S66SNGYB=GS1.2.1745757401.1.1.1745757406.0.0.0; _jzqy=1.1745807040.1745807040.1.jzqsr=baidu.-; _ga_KJTRWRHDL1=GS1.2.1745813568.14.1.1745813588.0.0.0; _ga_QJN1VP0CMS=GS1.2.1745813568.14.1.1745813588.0.0.0; _jzqckmp=1; sensorsdata2015jssdkcross=%7B%22distinct_id%22%3A%221966c53895a22e7-052e254f7112cd-482a5c03-1821369-1966c53895b249c%22%2C%22%24device_id%22%3A%221966c53895a22e7-052e254f7112cd-482a5c03-1821369-1966c53895b249c%22%2C%22props%22%3A%7B%22%24latest_traffic_source_type%22%3A%22%E7%9B%B4%E6%8E%A5%E6%B5%81%E9%87%8F%22%2C%22%24latest_referrer%22%3A%22%22%2C%22%24latest_referrer_host%22%3A%22%22%2C%22%24latest_search_keyword%22%3A%22%E6%9C%AA%E5%8F%96%E5%88%B0%E5%80%BC_%E7%9B%B4%E6%8E%A5%E6%89%93%E5%BC%80%22%7D%7D; _ga_LRLL77SF11=GS1.2.1745834391.5.1.1745834414.0.0.0; _ga_GVYN2J1PCG=GS1.2.1745834391.5.1.1745834414.0.0.0; lianjia_ssid=42fcdc54-30de-49b6-8fa8-2a321e098ae5; Hm_lvt_46bf127ac9b856df503ec2dbf942b67e=1745742241,1745807040,1745843663,1745892243; HMACCOUNT=813EC83E608DCA42; _jzqc=1; _ga_WLZSQZX7DE=GS1.2.1745892246.7.0.1745892246.0.0.0; _ga_TJZVFLS7KV=GS1.2.1745892246.7.0.1745892246.0.0.0; _ga_654P0WDKYN=GS1.2.1745892260.4.1.1745892269.0.0.0; _jzqx=1.1745576034.1745904799.13.jzqsr=hip%2Elianjia%2Ecom|jzqct=/.jzqsr=gz%2Elianjia%2Ecom|jzqct=/; select_city=440300; _qzjc=1; _ga_C4R21H79WC=GS1.2.1745904815.3.1.1745904860.0.0.0; Hm_lpvt_46bf127ac9b856df503ec2dbf942b67e=1745912126; _jzqa=1.2910519118316143600.1745574136.1745904799.1745912126.22; _qzja=1.415244424.1745748520942.1745904801192.1745912126305.1745904848094.1745912126305.0.0.0.12.4; _qzjto=4.2.0; _jzqb=1.1.10.1745912126.1; _qzjb=1.1745912126305.1.0.0.0; srcid=eyJ0Ijoie1wiZGF0YVwiOlwiMGE4OWJmZjk0ZmExODA3MmJlZTMxYTg3ODY4YjFhNjY4ODljNDQ2NTg4NjJkYzQ5MzBjNGNlODgxYjdjZTQ4NTc2YjU1MDdmNTYzNzc3NmU5YmQxMGZjYjY1ODAzZjNjNmQ0ZDAxNjBhNzk0YzM5OGYyYmNhNDU2MjhmZTYzYWJjNTEyMmQ5MGIzMjcwZGU2ZTNjOWYyNDcwZTM4OWIwOTEyZjRiOGQxNDYwYzIwMTM0OTZiM2MxMjAwZDUyZmNmMDAwOTBjYzUzMWU4ZDdkZGZlYzk2ZTdiYWJjODA1YzYzNjM4YmZiOGYwOGNjYjE5OTAwNTEyMmQ3NDI2YTFhMFwiLFwia2V5X2lkXCI6XCIxXCIsXCJzaWduXCI6XCI5NTAxODA4MVwifSIsInIiOiJodHRwczovL3N6LmxpYW5qaWEuY29tL2Vyc2hvdWZhbmcvcGc1MS8iLCJvcyI6IndlYiIsInYiOiIwLjEifQ=='
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 SLBrowser/9.0.6.2081 SLBChan/115 SLBVPV/64-bit',
            'Cookie': cookie.encode("utf-8").decode("latin1")
        }

        response = requests.get(url=url, headers=headers)  # get的请求工具
        # print(response.text)

        # 解析数据 提取想要的内容:re css xpath parser
        soup = BeautifulSoup(response.text, 'html.parser')
        # 第一次提取，提取所有li标签内容，返回列表（列表里面是selector对象）
        lis = soup.select('.sellListContent li')
        # print(lis)
        for li in lis:
            title_tag = li.select('.title a')
            if title_tag:
                try:
                    # 提取详情页
                    href = title_tag[0].get('href')  # 使用 get 方法获取 href 属性
                    detail_soup = crawl_detail_page(href, headers)

                    # print(href)
                    # 提取标题
                    title = title_tag[0].text.strip()  # 获取第一个匹配的标签的文本，并去除空白字符
                    # detail_soup['标题'] = title

                    # 提取地区
                    region_list = li.select('.flood a')
                    regions = [region.text for region in region_list]
                    region = '-'.join(regions)

                    # 提取房屋信息
                    house_info = li.select('.houseInfo')[0].text.split('|')
                    # print(house_info)
                    '''
                    num_rooms = house_info[0]  # 几室几厅
                    area = house_info[1]  # 房屋面积
                    orientation = house_info[2]  # 房屋朝向
                    renovation = house_info[3]  # 装修
                    floor = house_info[4]  # 楼层
                    build_time = house_info[5]  # 建立时间
                    house_type = house_info[6]  # 户型
                    '''
                    num_rooms, area, orientation, renovation, floor, build_time, house_type = extract_house_info(house_info)
                    #print(num_rooms, area, orientation, renovation, floor, build_time, house_type )

                    # 提取关注人数信息
                    follow_info = li.select('.followInfo')[0].text.split(' / ')
                    person_num = follow_info[0]  # 关注人数信息
                    update_build_time = follow_info[1]  # 发布日期信息

                    # 提取标签信息
                    tag_info = li.select('.tag span')
                    # tags = [tag.text for tag in tag_info]
                    # tag = '-'.join(tags)
                    vr_info, taxfree_info, subway_info, five_info, haskey_info = extract_tag_info(tag_info)

                    # 提取单价和总价
                    total_price = li.select('.totalPrice span')[0].text +'万'
                    unit_price = li.select('.unitPrice span')[0].text
                    # print(unit_price)
                    detail_soup['标题'] = title
                    detail_soup['总价'] = total_price
                    detail_soup['单价'] = unit_price
                    detail_soup['地区'] = region
                    detail_soup['几室几厅'] = num_rooms
                    detail_soup['房屋面积'] = area
                    detail_soup['楼层'] = floor,
                    detail_soup['建立时间'] = build_time

                    detail_soup['户型'] = house_type
                    detail_soup['关注人数'] = person_num
                    detail_soup['发布日期'] = update_build_time
                    detail_soup['VR看装修'] = vr_info

                    detail_soup['房本满五年'] = taxfree_info
                    detail_soup['近地铁'] = subway_info
                    detail_soup['房本满两年'] = five_info
                    detail_soup['随时看房'] = haskey_info
                    detail_soup['详情页'] = href

                    print(detail_soup)
                    try:
                        csv_writer.writerow(detail_soup)
                    except Exception as e:
                        print(f"数据写入失败: {str(e)}")
                    # print(title, region, num_rooms, area, orientation, renovation, floor, build_time, house_type, person_num, update_build_time, vr_info, taxfree_info, subway_info, five_info, haskey_info,total_price, unit_price, sep='|')
                except:
                    pass


