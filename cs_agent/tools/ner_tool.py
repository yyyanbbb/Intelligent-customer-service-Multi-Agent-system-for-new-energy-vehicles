"""
NER 工具：规则 + 正则抽取汽车领域实体，覆盖主流新能源品牌车型。
"""
from __future__ import annotations
import re

# 车型列表（长度降序，优先匹配最长）
VEHICLE_MODELS = [
    # 小鹏
    "MONA M03 Pro", "MONA M03", "MONA",
    "X9", "G9", "G6", "G3i", "G3",
    "P7\\+", "P7i", "P7", "P5",
    # 理想
    "理想 MEGA", "理想MEGA",
    "理想 L9", "理想L9",
    "理想 L8", "理想L8",
    "理想 L7", "理想L7",
    "理想 L6", "理想L6",
    "理想 i6", "理想i6",
    # 蔚来
    "ET9", "ET7", "ET5T", "ET5",
    "ES8", "ES7", "ES6",
    "EC7", "EC6", "EL6", "EL7",
    # 比亚迪
    "汉EV", "汉DM-i", "汉DM",
    "唐DM-i", "唐EV",
    "宋Pro DM-i", "宋Pro",
    "宋L", "海豹", "海豚",
    "海狮06", "海豹06GT", "海豹06",
    "海鸥", "元PLUS", "元UP",
    "腾势D9", "腾势N7", "腾势Z9",
    "仰望U9", "仰望U8",
    # 问界/AITO
    "问界M9", "问界M7", "问界M5",
    "AITO M9", "AITO M7", "AITO M5",
    # 极氪
    "极氪001", "极氪007", "极氪7X",
    "极氪MIX", "极氪009", "ZEEKR 001",
    # 小米汽车
    "SU7 Ultra", "SU7 Pro", "SU7",
    "YU7",
    # 特斯拉
    "Model 3", "Model Y", "Model S", "Model X",
    "Cybertruck",
    # 零跑
    "零跑C16", "零跑C11", "零跑C10",
    "零跑B01", "零跑T03",
    # 深蓝
    "深蓝G318", "深蓝S07", "深蓝S05",
    # 阿维塔
    "阿维塔12", "阿维塔11",
    # 岚图
    "岚图追光", "岚图FREE", "岚图梦想家",
    # 吉利银河
    "银河E8", "银河E5", "银河L7",
    # 埃安/昊铂
    "昊铂SSR", "昊铂GT",
    "AION Y Plus", "AION Y", "AION V", "AION S", "AION LX",
    # 智己
    "智己LS6", "智己L6",
    # 长安启源
    "启源Q07", "启源A05",
    # 极狐
    "阿尔法S5", "阿尔法T",
    # 哈弗/坦克/魏牌
    "坦克400", "坦克300",
    "哈弗猛龙", "哈弗H6",
    "蓝山DHT",
    # 极越
    "极越07",
    # 五菱
    "五菱缤果", "五菱星光",
    "宏光MINIEV",
]

COMPONENT_KWS = [
    "电池", "电机", "电控", "空调", "轮胎", "刹车", "制动",
    "车机", "屏幕", "座椅", "天窗", "车窗", "雨刮", "大灯",
    "充电桩", "充电口", "充电枪", "悬架", "转向", "方向盘",
    "激光雷达", "摄像头", "超声波", "BMS", "OBC",
    "云辇", "空气悬架", "CDC减震", "刀片电池", "麒麟电池",
]

FAULT_KWS = [
    "不制冷", "不制热", "黑屏", "死机", "异响", "漏水", "抖动",
    "充不进", "打不开", "关不上", "扎钉", "爆胎", "漏气",
    "报警", "失灵", "卡顿", "断连", "掉电", "不启动",
    "异味", "起火", "冒烟", "过热", "刮蹭", "剐蹭",
]

FEATURE_KWS = [
    "XNGP", "NGP", "APA", "VPA", "LCC", "ACC",
    "ADS", "NIO Pilot", "NAD", "NOA",
    "智驾", "辅助驾驶", "自动泊车", "记忆泊车",
    "城市NOA", "城市NCA", "高速NOA",
    "语音", "OTA", "超充", "800V", "V2L", "NFC",
    "换电", "超级快充", "5C快充", "4C快充",
    "鸿蒙", "澎湃OS", "银河N OS", "天玑XOS",
]

BRAND_KWS = [
    "比亚迪", "小鹏", "理想", "蔚来", "问界", "AITO",
    "特斯拉", "小米汽车", "极氪", "零跑", "深蓝",
    "阿维塔", "岚图", "仰望", "腾势", "智己",
    "吉利银河", "银河", "埃安", "昊铂", "广汽",
    "长安", "哈弗", "坦克", "魏牌", "奇瑞",
    "极越", "极狐", "五菱", "Polestar", "极星",
]

BUDGET_PATTERN = re.compile(r"(\d+[\.\d]*)\s*万")


def extract_entities(text: str) -> list[dict]:
    """规则 + 正则抽取汽车领域实体。"""
    entities: list[dict] = []
    seen: set[tuple[int, int]] = set()

    def _add(match_text: str, label: str, start: int, end: int):
        if (start, end) not in seen:
            seen.add((start, end))
            entities.append({"text": match_text, "label": label, "start": start, "end": end})

    # 车型（按长度降序，优先匹配最长）
    for pat in VEHICLE_MODELS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            _add(m.group(), "vehicle_model", m.start(), m.end())

    # 品牌
    for kw in BRAND_KWS:
        idx = text.find(kw)
        while idx != -1:
            _add(kw, "brand", idx, idx + len(kw))
            idx = text.find(kw, idx + 1)

    # 预算
    for m in BUDGET_PATTERN.finditer(text):
        _add(m.group(), "budget", m.start(), m.end())

    # 部件、故障、功能
    for kw_list, label in [
        (COMPONENT_KWS, "component"),
        (FAULT_KWS, "fault"),
        (FEATURE_KWS, "feature"),
    ]:
        for kw in kw_list:
            idx = text.find(kw)
            while idx != -1:
                _add(kw, label, idx, idx + len(kw))
                idx = text.find(kw, idx + 1)

    return entities
