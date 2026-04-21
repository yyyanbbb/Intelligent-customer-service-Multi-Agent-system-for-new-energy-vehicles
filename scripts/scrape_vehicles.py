"""
汽车数据爬虫 v8
策略：
1. 收集汽车之家热门新能源车 ID（/energy/ /newenergy/ 页面）
2. 访问每个车系页，滚动触发懒加载，从标题提取真实品牌/车型名
3. 官网直连获取更多参数（价格过滤到合理范围 5-300万）
4. 后处理：过滤非新能源车，保留真实有效数据
"""
from __future__ import annotations
import asyncio, json, re, sys
from pathlib import Path
from collections import Counter
from playwright.async_api import async_playwright, Page, Response

sys.stdout.reconfigure(encoding='utf-8')

OUT_DIR = Path(__file__).parent.parent / "cs_agent" / "knowledge"
OUT_DIR.mkdir(parents=True, exist_ok=True)

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"

# 目标品牌（用于过滤/归类）
TARGET_BRANDS = {
    "小鹏", "比亚迪", "特斯拉", "理想", "问界", "蔚来", "极氪",
    "小米汽车", "小米", "零跑", "深蓝", "腾势", "阿维塔", "岚图",
    "智界", "方程豹", "仰望", "哪吒", "埃安", "昊铂", "吉利银河",
    "奇瑞", "奇瑞新能源", "长安", "长安启源", "东风", "华为",
}

# 官网直连车型列表
OFFICIAL_MODELS = [
    ("小鹏", "G6", "https://www.xiaopeng.com/g6/"),
    ("小鹏", "G9", "https://www.xiaopeng.com/g9/"),
    ("小鹏", "X9", "https://www.xiaopeng.com/x9/"),
    ("小鹏", "P7+", "https://www.xiaopeng.com/p7plus/"),
    ("小鹏", "MONA M03", "https://www.xiaopeng.com/mona-m03/"),
    ("蔚来", "ES6", "https://www.nio.cn/es6"),
    ("蔚来", "ET5", "https://www.nio.cn/et5"),
    ("蔚来", "ET7", "https://www.nio.cn/et7"),
    ("蔚来", "ES8", "https://www.nio.cn/es8"),
    ("蔚来", "EC6", "https://www.nio.cn/ec6"),
    ("蔚来", "ET5T", "https://www.nio.cn/et5t"),
    ("极氪", "001", "https://www.zeekrlife.com/zeekr001"),
    ("极氪", "007", "https://www.zeekrlife.com/zeekr007"),
    ("极氪", "X", "https://www.zeekrlife.com/zeekrx"),
    ("极氪", "Mix", "https://www.zeekrlife.com/zeekrmix"),
    ("理想", "L9", "https://www.lixiang.com/l9.html"),
    ("理想", "L8", "https://www.lixiang.com/l8.html"),
    ("理想", "L7", "https://www.lixiang.com/l7.html"),
    ("理想", "L6", "https://www.lixiang.com/l6.html"),
    ("理想", "MEGA", "https://www.lixiang.com/mega.html"),
    ("问界", "M9", "https://www.aito.com/m9"),
    ("问界", "M7", "https://www.aito.com/m7"),
    ("问界", "M5", "https://www.aito.com/m5"),
    ("哪吒", "S", "https://www.neta-auto.com/s"),
    ("哪吒", "X", "https://www.neta-auto.com/x"),
    ("零跑", "C11", "https://www.leapmotor.com/c11"),
    ("零跑", "C10", "https://www.leapmotor.com/c10"),
    ("零跑", "B10", "https://www.leapmotor.com/b10"),
]

# autohome 热门新能源车 ID（直接收集所有出现过的热门 ID）
# 后续从页面标题自动识别品牌
KNOWN_EV_SERIES: list[int] = [
    # 比亚迪系列
    7588, 7851, 8087, 6897, 7356, 7177, 6898, 7450, 7601, 7730,
    # 比亚迪更多
    7163, 7066, 6988, 7073, 7859,
    # 问界/华为
    7444, 7291, 7600, 7831, 7950,
    # 蔚来
    6308, 7027, 6307, 6309, 7386,
    # 极氪
    6934, 7430, 7270, 7859,
    # 小米
    7900, 8082, 8180,
    # 零跑
    6660, 6924, 7612, 7980,
    # 小鹏
    6272, 7796, 7658, 7329,
    # 理想
    6986, 7135, 7383, 7622, 7461,
    # 其他热门新能源
    7356, 6898, 7177, 7588, 7650, 8100, 7864, 8512, 5770, 8087,
    7578, 5279, 5964, 7918, 7452,
    # 更多
    7208, 6985, 7664, 7856, 7113, 7278,
]
KNOWN_EV_SERIES = list(dict.fromkeys(KNOWN_EV_SERIES))  # 去重


def parse_price(text: str) -> str:
    """提取合理价格区间（5-300万）"""
    matches = re.findall(r"(\d{1,3}\.?\d*)\s*万", text)
    valid = []
    for m in matches:
        v = float(m)
        if 5 <= v <= 300:
            valid.append(v)
    if valid:
        mn = min(valid)
        return f"{mn}万元起"
    return ""


def extract_params(text: str) -> dict:
    params = {}
    patterns = [
        (r"(?:CLTC|纯电续航|综合续航)[^0-9]{0,15}(\d{3,4})\s*(?:公里|km)", "CLTC续航km"),
        (r"(\d{3,4})\s*(?:公里|km)[^，。\n]{0,5}CLTC", "CLTC续航km"),
        (r"(\d{2,3}\.?\d*)\s*kWh", "电池容量kWh"),
        (r"0[-–]?100[^0-9]{0,8}(\d+\.?\d+)\s*[秒s]", "百公里加速s"),
        (r"(\d+\.?\d+)\s*[秒s][内]?[破百]", "百公里加速s"),
        (r"轴距[：:\s]*(\d{4})\s*mm", "轴距mm"),
        (r"最高[车]?速[：:]\s*(\d{3})\s*(?:km|公里)", "最高车速kmh"),
        (r"最大功率[：:]\s*(\d{2,4})\s*kW", "最大功率kW"),
        (r"最大扭矩[：:]\s*(\d{3,4})\s*N", "最大扭矩Nm"),
        (r"整备质量[：:]\s*(\d{4})\s*kg", "整备质量kg"),
        (r"快充[^0-9]{0,8}(\d+)\s*分钟", "快充时间min"),
    ]
    for pat, key in patterns:
        if key not in params:
            m = re.search(pat, text)
            if m:
                params[key] = m.group(1).strip()
    return params


def parse_brand_model_from_title(title: str) -> tuple[str, str]:
    """
    从汽车之家标题解析品牌和车型。
    格式: 【车型】品牌_车型报价_车型图片_汽车之家
    或: 品牌 车型 报价
    """
    # 格式1: 【MODEL】BRAND_MODEL报价...
    m = re.match(r'【([^】]+)】([^_]+)_', title)
    if m:
        model = m.group(1).strip()
        brand = m.group(2).strip()
        return brand, model

    # 格式2: BRAND MODEL 报价...
    parts = title.split('_')
    if len(parts) >= 2:
        return parts[0].strip(), parts[1].strip()

    return "", title


async def _scroll_and_wait(page: Page, ms: int = 3000) -> None:
    """滚动页面触发懒加载内容"""
    await page.evaluate("""async () => {
        for (let i = 0; i < 5; i++) {
            window.scrollBy(0, window.innerHeight);
            await new Promise(r => setTimeout(r, 400));
        }
        window.scrollTo(0, 0);
    }""")
    await page.wait_for_timeout(ms)


async def scrape_autohome_series(page: Page, series_id: int) -> dict | None:
    """抓取汽车之家车系页，自动识别品牌"""
    url = f"https://www.autohome.com.cn/{series_id}/"
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=25000)
        await _scroll_and_wait(page, 3000)

        title = await page.title()
        if not title or any(x in title for x in ["404", "找不到", "出错", "error"]):
            return None

        brand, model = parse_brand_model_from_title(title)
        if not brand:
            return None

        body_text = await page.evaluate("() => document.body.innerText.slice(0, 12000)")
        specs = extract_params(body_text)
        price = parse_price(body_text)

        # 尝试参数子页（/param/ 或 /config/ 路径）
        spec_links = await page.evaluate("""() =>
            Array.from(document.querySelectorAll('a[href]'))
                .map(a => ({href: a.href, text: a.innerText.trim()}))
                .filter(a => /参数|配置|规格/.test(a.text) && /\\/\\d{4,}/.test(a.href))
                .slice(0, 3)
        """)
        for lnk in spec_links:
            h = lnk["href"]
            if h.startswith("http") and str(series_id) in h:
                try:
                    await page.goto(h, wait_until="domcontentloaded", timeout=15000)
                    await _scroll_and_wait(page, 2500)
                    st = await page.evaluate("() => document.body.innerText.slice(0, 10000)")
                    specs.update(extract_params(st))
                    tbl = await page.evaluate("""() => {
                        const r = {};
                        document.querySelectorAll('tr').forEach(row => {
                            const c = row.querySelectorAll('td,th');
                            if (c.length >= 2) {
                                const k = c[0].innerText.trim();
                                const v = c[1].innerText.trim();
                                if (k && v && k !== v && k.length < 25) r[k] = v;
                            }
                        });
                        return r;
                    }""")
                    specs.update(tbl)
                    break
                except:
                    pass

        print(f"    [{series_id}] {brand} {model} | 参数:{len(specs)} | 价格:{price}")
        return {
            "brand": brand, "series_id": series_id,
            "name": f"{brand} {model}", "price": price,
            "specs_raw": specs, "source_url": url,
            "body_text": body_text[:1500],
        }
    except Exception as e:
        print(f"    [{series_id}] 错误: {e}")
        return None


async def collect_more_series_ids(page: Page) -> list[int]:
    """从汽车之家热门榜单/新能源专区收集更多 ID"""
    collected = []

    # 热门新能源车榜单
    urls_to_scan = [
        "https://www.autohome.com.cn/rank/",
        "https://www.autohome.com.cn/energy/",
        "https://www.autohome.com.cn/newenergy/",
        "https://new.autohome.com.cn/newenergy/",
    ]

    for url in urls_to_scan:
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=20000)
            await page.wait_for_timeout(3000)
            html = await page.content()
            # 从 /XXXX/ 格式链接提取 ID
            ids = re.findall(r'href="https://www\.autohome\.com\.cn/(\d{4,5})/', html)
            ids += re.findall(r'href="/(\d{4,5})/"', html)
            for sid in ids:
                sid = int(sid)
                if 3000 < sid < 15000 and sid not in collected:
                    collected.append(sid)
            print(f"  {url}: +{len(ids)} IDs")
        except Exception as e:
            print(f"  {url}: {e}")

    return collected


async def scrape_official(page: Page, brand: str, model: str, url: str) -> dict | None:
    """抓取官网车型页"""
    try:
        await page.goto(url, wait_until="networkidle", timeout=35000)
        await _scroll_and_wait(page, 3000)
        body = await page.evaluate("() => document.body.innerText")
        if len(body) < 300:
            return None
        specs = extract_params(body)
        price = parse_price(body)
        print(f"  {brand} {model}: 参数={len(specs)} 价格={price}")
        return {
            "brand": brand, "name": f"{brand} {model}",
            "price": price, "specs_raw": specs,
            "source_url": url, "body_text": body[:2000],
        }
    except Exception as e:
        print(f"  {brand} {model} 失败: {e}")
        return None


async def main():
    print("=" * 60)
    print("汽车数据爬虫 v8 — 滚动懒加载 + 官网直连")
    print("=" * 60)

    all_raw = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--lang=zh-CN"],
        )
        ctx = await browser.new_context(
            user_agent=UA, locale="zh-CN",
            viewport={"width": 1280, "height": 900},
        )
        await ctx.route("**/*.{woff,woff2,ttf,otf,mp4,webm}", lambda r: r.abort())
        page = await ctx.new_page()

        # Step 1: 收集更多车系 ID
        print("\n=== Step 1: 从热门榜单收集 ID ===")
        extra_ids = await collect_more_series_ids(page)
        all_ids = list(dict.fromkeys(KNOWN_EV_SERIES + extra_ids))
        print(f"  总计 {len(all_ids)} 个车系 ID")

        # Step 2: 抓取所有车系页，自动识别品牌
        print("\n=== Step 2: 抓取车系页（自动识别品牌）===")
        for sid in all_ids:
            d = await scrape_autohome_series(page, sid)
            if d:
                all_raw.append(d)
            await asyncio.sleep(0.7)

        # Step 3: 官网直连
        print("\n=== Step 3: 品牌官网 ===")
        for brand, model, url in OFFICIAL_MODELS:
            d = await scrape_official(page, brand, model, url)
            if d:
                all_raw.append(d)
            await asyncio.sleep(1.0)

        await browser.close()

    # 保存
    raw_path = OUT_DIR / "raw_scraped.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_raw, f, ensure_ascii=False, indent=2)
    print(f"\n原始数据: {raw_path} ({len(all_raw)} 条)")

    vehicles = []
    for d in all_raw:
        v = {
            "brand": d.get("brand", ""),
            "model": d.get("name", ""),
            "price": d.get("price", ""),
            "specs": d.get("specs_raw", {}),
            "source_url": d.get("source_url", ""),
            "summary": d.get("body_text", "")[:500],
        }
        if v["brand"] and (v["model"] or v["price"] or v["specs"]):
            vehicles.append(v)

    vehicles_path = OUT_DIR / "vehicles.json"
    with open(vehicles_path, "w", encoding="utf-8") as f:
        json.dump(vehicles, f, ensure_ascii=False, indent=2)
    print(f"结构化数据: {vehicles_path} ({len(vehicles)} 条)")

    print("\n=== 品牌统计 ===")
    for b, c in Counter(v["brand"] for v in vehicles).most_common():
        print(f"  {b}: {c} 条")


if __name__ == "__main__":
    asyncio.run(main())
