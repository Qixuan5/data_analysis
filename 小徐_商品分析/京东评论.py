import csv
import random
import re
import time
import requests
from DrissionPage import Chromium
from DrissionPage import ChromiumOptions
from urllib.parse import quote_plus  

def driver_login(tab, url):
    tab.get(url, retry=5, interval=3)
    input('请手动登录后按回车继续(如今天已经登录过那么直接回车即可).......')


def drop_down(tab):
    init_count = 0
    for i in range(13):
        init_count += 300
        js = f'document.documentElement.scrollTop={init_count}'
        tab.run_js(js)
        time.sleep(random.uniform(0.3, 0.6))
    time.sleep(random.uniform(0.5, 1))
count = 0
def driver_get_data(tab, sku):
    global count
    url=f'https://item.jd.com/{sku}.html'
    tab.get(url, retry=5, interval=3)
    time.sleep(1)
    selector = f'css:.price.J-p-{sku}'
    price_text = tab.ele(selector).text.strip()
    tab.listen.start(
        'https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments&client=pc')
    tab.listen.set_targets(
        targets='https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments&client=pc', method='GET')
    tab.ele('text:商品评价', timeout=0.5).click(by_js=True)

    new_tab = browser.latest_tab     

    if new_tab!=tab:                                    # 真开了新页

        new_tab.close()    
        input("manual fix")                    # 关闭
    time.sleep(2)
    for packet in tab.listen.steps():
        json_data = packet.response.body

        data_list = json_data.get('comments')
        for item in data_list:
            try:
                location = item.get('location')  # 评论IP
                if location==None:
                    continue
                nickname = item.get('nickname')  # 用户名
                productColor = item.get('productColor')  # 购买颜色
                productSize = item.get('productSize')  # 型号
                plusAvailable = item.get('plusAvailable')  # 是否是会员
                if plusAvailable == 201:
                    is_plus = 1
                else:
                    is_plus = 0
                buyCount = item.get('extMap').get('buyCount')  # 购买次数
                score = item.get('score')  # 评论分数
                creationTime = item.get('creationTime')  # 评论时间
                usefulVoteCount = item.get('usefulVoteCount')  # 赞成数
                content = item.get('content')  # 评论内容
                imageCount = item.get('imageCount')  # 是否图片评论
                if imageCount:
                    imageCount_res = 1
                else:
                    imageCount_res = 0
                print([sku,price_text,nickname, productColor, productSize, is_plus, buyCount, score, creationTime,
                    location, usefulVoteCount, content, imageCount_res])
                csv_writer.writerow([sku,price_text,nickname, productColor, productSize, is_plus, buyCount, score, creationTime,
                                    location, usefulVoteCount, content, imageCount_res])
                count += 1

            except Exception:
                print('报错了......')

        print('总计数:', count)

        next_label = tab.ele('c=.com-table-footer .ui-pager-next', timeout=0.5)
        if next_label and next_label.text == '下一页':
            time.sleep(random.uniform(2, 4))
            next_label.click(by_js=True)
        else:
            break
    return None

import time, random, requests
from urllib.parse import quote_plus  

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"),
    "Referer": "https://www.jd.com/",
    "Cookie": "__jdv=122270672%7Cdirect%7C-%7Cnone%7C-%7C1745742590315; __jdu=1745742590315476553045; pinId=tOIy--EudxqY8v5dHZyNr7V9-x-f3wj7; pin=jd_708097f895d84; unick=jd_zy2gy655fhg5uy; _tp=hnWoqZgh40SF%2BaY3V9lkMT7smig6Z%2BUT1XumnqdYLgU%3D; _pst=jd_708097f895d84; shshshfpa=6a0ee20b-4f91-7e10-8553-61136c406ac0-1745742610; shshshfpx=6a0ee20b-4f91-7e10-8553-61136c406ac0-1745742610; areaId=2; rkv=1.0; qrsc=3; ipLoc-djd=2-2825-61086-0; TrackID=1XKXDkmUoRnFuLAbEGtLPijPTHzSIF3yGLHewnF8x3PCrzqS7UZeSo9NRQtt4xYxMGuZiI-EYMKKJiO-htAKJLkDERx_WhY_FhYnQKLkaCY4VGb1hN3VlJSr5vqDT35FQ; thor=4A0C02E4453E2C70A63DDF49072EB58F8D08A8F2EFA648FECC16C48CB38563BB6C14BE7033B6ECF13C9CD8FEDF199E6D053BA0F4C7A6EB63A2263CF7D9A3C465FA8836C47672E2F95906584AC941DC715C873422CD896DAB509C8C15B5DA9B31DE5CF1DB67E4301BE702BC5A571F6ED832469A8D1E5C9083D976E20F95BDB956AFE1EC1791471F003591EE45639629645506BB7C7113916554E9841A13A9DD20; light_key=AASBKE7rOxgWQziEhC_QY6yaC25a_07sWtUwKXjYuvKSdNSlQ3rxmZA5Yo5MjyJ4hjYF5T2-; ceshi3.com=000; PCSYCityID=CN_370000_370600_0; xapieid=jdd03AATKT35XAJ3I7VUCPK7JYJQ4ET3QPQFD6F2YIU7B642TY2AKWMP5HWQ6NOXKOVVN25TS4J76E5CNKLFDP4ZBBZHXJEAAAAMWTQWPLKIAAAAACTZZH4XFPY4K7EX; 3AB9D23F7A4B3C9B=AATKT35XAJ3I7VUCPK7JYJQ4ET3QPQFD6F2YIU7B642TY2AKWMP5HWQ6NOXKOVVN25TS4J76E5CNKLFDP4ZBBZHXJE; retina=1; cid=9; appCode=ms0ca95114; mba_muid=1745742590315476553045; webp=1; visitkey=7907988837712504355; PPRD_P=UUID.1745742590315476553045; sc_width=1470; flash=3_Q6PelWeXHoW2-wrdv6OZ2qFrLNDFTGz3yXwtStjFpecI1utf73vlgc1tTu-DbvEWlavykMSLjhvPCabS5M-i99tGZGJ_A8Ud2xH18N2Fcq-zrkeYxRaBoZNFMvNIlh74yLHUCBwZhVTaL3yqMTiR-YZDIdsM6pk1dzlJCkJx8j7w1pFaS_htZe**; TrackerID=vuU9EjXEw7kQ-l8eXIIfaI_RgWtSpbCxI8KHGviTzVuZ9YElKsWAPE-8CmaV6yCK-2pS_Fpax0RROUC6TRqxmJp_ZF977exlrrcb8C_WA7FFXC4TLq4hZGdE1PKXaiHk; pt_key=AAJoF5vOADDfjJIkn9QBMdIZzVrIOuT-20PwaSWt3AOZmSCmipn4CNr1Bk4E7dDNeQciMAyiyTo; pt_pin=jd_708097f895d84; pt_token=auwmxlb4; pwdt_id=jd_708097f895d84; sfstoken=tk01md6bc1d1da8sMSszKzIrM3gxY83BC/fsMuZx0zfOEhmlvDZcNXK6fONuGrgZq67z6YcukI5pz5peLf7YWnUksC7M; __jd_ref_cls=MDownLoadFloat_ApiDownloadAppLayerConfigData; 3AB9D23F7A4B3CSS=jdd03AATKT35XAJ3I7VUCPK7JYJQ4ET3QPQFD6F2YIU7B642TY2AKWMP5HWQ6NOXKOVVN25TS4J76E5CNKLFDP4ZBBZHXJEAAAAMWTRCX6DYAAAAADONEWM3PJLS7ZQX; __jda=181111935.1745742590315476553045.1745742590.1746369239.1746377634.9; __jdc=181111935; shshshfpb=BApXSV65On_NA-EwbvdaQpDeaCGTXhtsaBgc1IT9t9xJ1ItZfQtGGykqy2y2iatN3K-EuPcyPsg; pt_st=1_UOAMRaWlwPgNL1gnY_W53Y3gY93DrQz2ZtuSYtjAMMDrZZ_gob-XsdRLSAphovrC86LnuAprqzoeUDK5l3wUkMsda9i-l1Qx2uoLajJgfvQZLc1yNUfNASJPmd_1N73s2DFgpw9J-InuCjmdA4JA2mIyjGPl-V0bU1dm6Jz9uvsF7-PPj3LOx4Z5qt2p2XM87bpgcghyjh0wHIhyOKmurfXpPOnowafXYdPzLsBS"
}



def search_skus(keyword: str, pages: int = 3):
    PATTERN = re.compile(r'data-sku="(\d{6,})"')  
    kw = quote_plus(keyword)                  
    sku_all = set()

    for p in range(1, pages + 1):
        url = (f"https://search.jd.com/Search?keyword={kw}&enc=utf-8"
               f"&page={p*2-1}&log_id={time.time()}")   # 京东翻页逻辑：1,3,5…
        html = requests.get(url, headers=HEADERS, timeout=10).text

        # 抓 <li class="gl-item" data-sku="...">
        sku_all.update(PATTERN.findall(html))

        # 兜底：如果脚本里还有 "skuIds":"id1,id2"
        m = re.search(r'"skuIds":"([\d,]+)"', html)
        if m:
            sku_all.update(m.group(1).split(","))

        time.sleep(random.uniform(1, 2))       # 防止限速
    return list(sku_all)

if __name__ == '__main__':

    keyword=input("搜索关键字：")
    file = open(f'{keyword}-评论数据.csv', mode='a', encoding='utf-8-sig', newline='')
    csv_writer = csv.writer(file, delimiter=',')
    csv_writer.writerow(
        ['sku','价格','评论IP','用户名', '购买颜色', '型号', '是否是会员', '购买次数', '评论分数', '评论时间', '赞成数', '评论内容',
         '是否是图片评论'])

    path = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
    ChromiumOptions().set_browser_path(path).save()

    co = ChromiumOptions()
    co.set_pref(arg='profile.default_content_settings.popups', value='0')

    browser = Chromium(co)
    tab1 = browser.latest_tab

    login_flag = True #每一天开始需改成True
    if login_flag:
        driver_login(tab1, 'https://passport.jd.com/uc/login')
        print('登录完成......')
    skus=search_skus(keyword, pages=1)
    for sku in skus:
        driver_get_data(tab1, sku)

    tab1.close()
    browser.quit()
