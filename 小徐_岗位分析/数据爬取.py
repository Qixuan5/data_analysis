import requests
import parsel
import csv
import time
import random
from urllib.parse import urljoin

base_url = "https://sz.58.com/{area}/quanzhizhaopin/"
areas = ['baoan', 'nanshan']  # 可添加更多区域

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.6261.95 Safari/537.36',
    'Cookie':'f=n; commontopbar_new_city_info=4%7C%E6%B7%B1%E5%9C%B3%7Csz; commontopbar_ipcity=sh%7C%E4%B8%8A%E6%B5%B7%7C0; id58=OSXcz2gPX/k4cKfvA5xnAg==; wmda_uuid=614ddc1287762e74f18ad7aafa0c943f; wmda_new_uuid=1; wmda_visited_projects=%3B1731916484865; 58tj_uuid=57882230-2b02-445c-9e80-f1debe80ecf9; als=0; sessionid=9097e000-e0b8-48c8-9f28-cee290d6c0d4; fzq_h=6fad3dc7336ad695192ed44648e5be44_1746197737365_d477c19ea6444fb495176c27564e92be_47901739930148925548327397250483467344; new_uv=3; utm_source=; spm=; init_refer=; Hm_lvt_5bcc464efd3454091cf2095d3515ea05=1745838075,1746197740; HMACCOUNT=5D82BB15C7EE2616; new_session=0; qz_gdt=; Hm_lvt_b2c7b5733f1b8ddcfc238f97b417f4dd=1745838934,1746197768; xxzlclientid=9f8d4dbc-ed60-4aef-a7f8-1746197964482; xxzlxxid=pfmxsdpm+ZJIj0rHQPqcSo+lIlklJc1lF+9O/3nKtJXZNmp2nNVhponiWqE2HejFIj77; xxzlbbid=pfmbM3wxMDI3M3wxLjEwLjB8MTc0NjE5Nzk2NDk2MjAxMjM5MHwxSmVFM202Rm5kb3NxcVdSUDhnMEVXRXF6QmNacXN6aTlhdkJLQVVOTm0wPXwwM2Q0Y2ZiNGM2MzllYmFlNDUwYjE5NWM5MjI3MzUxNF8xNzQ2MTk3OTYzMDU3X2RiYjZmMWM3MjEzODQ2N2VhYmVmZWY1ODI1MDY3ZjI5XzQ3OTAxNzM5OTMwMTQ4OTI1NTQ4MzI3Mzk3MjUwNDgzNDY3MzQ0fDMxNWY5NTJmNDU5Y2E1NWJkYjlkMGZkYjc1YmViNzY4XzE3NDYxOTc5NjM5MzVfMjU1; f=n; fzq_js_infodetailweb=18b32aabb5125a4ec85d3b27f272140e_1746198951459_6; Hm_lpvt_b2c7b5733f1b8ddcfc238f97b417f4dd=1746198952; ppStore_fingerprint=C9B2244672CFB52C90962973C1ADE19C2EFC0A473EB9A8E6%EF%BC%BF1746198954806; JSESSIONID=212DE84D49DE0576FA8CC63536E31C6A; wmda_report_times=25; fzq_js_zhaopin_list_pc=aab8b5ea021f9c3f066bf4d9eefc80ca_1746199157200_9; Hm_lpvt_5bcc464efd3454091cf2095d3515ea05=1746199157'
}

target_benefits = [
    "五险一金", "包住", "包吃", "年底双薪", "周末双休",
    "交通补助", "加班补助", "饭补", "话补", "房补"
]

results = []

for area in areas:
    for page in range(1, 2):  # 可增大页数
        if page == 1:
            url = base_url.format(area=area) + "?key=%E9%94%80%E5%94%AE%E4%B8%93%E5%91%98&cmcskey=%E9%94%80%E5%94%AE%E4%B8%93%E5%91%98&final=1&jump=1&specialtype=gls&classpolicy=LBGguide_A,main_B,job_B,hitword_false,uuid_aJhibZxne7MSzbKpxMmTdxtsH8haDGAG,displocalid_4,from_main,to_jump,tradeline_job,classify_A&search_uuid=aJhibZxne7MSzbKpxMmTdxtsH8haDGAG&search_type=suggest&pid=817173548212584448&PGTID=0d3002a2-0071-4c4b-9fc9-e9877870081a&ClickID=1"
        else:
            url = base_url.format(area=area) + f"pn{page}/?key=%E9%94%80%E5%94%AE%E4%B8%93%E5%91%98&cmcskey=%E9%94%80%E5%94%AE%E4%B8%93%E5%91%98&final=1&jump=1&specialtype=gls&classpolicy=LBGguide_A,main_B,job_B,hitword_false,uuid_aJhibZxne7MSzbKpxMmTdxtsH8haDGAG,displocalid_4,from_main,to_jump,tradeline_job,classify_A&search_uuid=aJhibZxne7MSzbKpxMmTdxtsH8haDGAG&search_type=suggest&pid=817173548212584448&PGTID=0d3002a2-0071-4c4b-9fc9-e9877870081a&ClickID=1"
        
        print(f"访问列表页：{url}")
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            selector = parsel.Selector(response.text)

            job_items = selector.css('ul#list_con > li.job_item.clearfix')
            for index, job in enumerate(job_items, start=1):
                href = job.css('div.job_name a::attr(href)').get()
                if not href:
                    continue
                detail_url = urljoin(url, href)
                print(f"  → 进入详情页（li位置: {index}）：{detail_url}")

                try:
                    detail_resp = requests.get(detail_url, headers=headers, timeout=10)
                    detail_resp.raise_for_status()
                    detail_sel = parsel.Selector(detail_resp.text)
                    jobs = selector.css('ul#list_con > li.job_item.clearfix')

                    for job in jobs:
                        job_salary = job.css('p.job_salary::text').get() or "无"
                        job_wel_list = job.css('div.job_wel.clearfix span::text').getall()
                        job_wel_list = [w.strip() for w in job_wel_list]

                        # 构造数据项
                        job_data = {'工资': job_salary}
                        for benefit in target_benefits:
                            job_data[benefit] = "有" if benefit in job_wel_list else ""

                        results.append(job_data)
                

                    time.sleep(random.uniform(0.8, 1.5))  # 防封

                except Exception as e:
                    print(f"❌ 详情页出错（跳过）：{detail_url} → {e}")
                    continue

            time.sleep(random.uniform(1.5, 2.5))  # 防封
        except Exception as e:
            print(f"❌ 列表页失败：{url} → {e}")
            continue

# 写入 CSV
fieldnames = ['工资'] + target_benefits
with open('jobs_详细信息.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print("✅ 数据已成功写入 jobs_详细信息.csv")
