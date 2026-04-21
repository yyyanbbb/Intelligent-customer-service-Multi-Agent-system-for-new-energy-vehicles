"""
路由节点。三级意图分类：规则关键词 → LoRA 分类器 → LLM tool-calling 兜底。
缓存命中时直接短路，跳过后续所有节点。
"""
from __future__ import annotations
import json
from cs_agent.state import CSState
from cs_agent.tools.ner_tool import extract_entities
from cs_agent.memory import update_from_turn, memory_as_context
from cs_agent.observability import cache_lookup, Timer
from cs_agent.llm_client import llm_chat, get_active_backend
from cs_agent.tools.lc_tools import INTENT_TOOL_SCHEMA

_INTENT_KEYWORDS: dict[str, list[str]] = {
    "roadside": [
        "救援", "拖车", "道路救援", "紧急", "抛锚", "趴窝", "出事故",
        "碰撞", "追尾", "路边", "打不着火", "无法启动", "求助",
    ],
    "complaint": [
        "投诉", "差评", "举报", "维权", "赔偿", "退款", "退车", "骗人",
        "虚假宣传", "质量问题", "不满意", "态度差", "不负责", "曝光",
        "315", "消协", "起诉", "律师", "理赔纠纷",
    ],
    "aftersales": [
        "故障", "坏了", "异响", "不工作", "维修", "保养", "质保",
        "黑屏", "不制冷", "不制热", "漏水", "报警", "异常", "抖动",
        "扎钉", "爆胎", "召回", "检修", "售后", "维保",
        "异味", "起火", "冒烟", "过热", "刮蹭", "失灵", "工单",
    ],
    "charging": [
        "充电", "充不进去", "充电慢", "充电桩", "超充站", "快充", "慢充",
        "充电价格", "充电费", "找充电站", "续航焦虑", "电量", "SOC",
        "V2L", "V2G", "充电线", "充电口", "充电适配", "800V充电",
    ],
    "order_tracking": [
        "订单", "交付", "几时交车", "什么时候到", "发货", "物流",
        "等待", "排队", "生产进度", "交付进度", "提车", "验车",
        "订车", "大定", "小定", "交车时间", "尾款", "交付中心",
    ],
    "insurance": [
        "保险", "理赔", "出险", "保费", "投保", "续保", "车险",
        "新能源险", "上牌", "牌照", "临牌", "交强险", "商业险",
        "三者险", "车损险", "定损", "报案",
    ],
    "test_drive": [
        "试驾", "预约试驾", "体验", "试乘", "预约看车", "到店",
        "展厅", "体验中心", "试驾活动", "预约时间", "门店",
    ],
    "account": [
        "账户", "账号", "登录", "密码", "注册", "APP", "应用",
        "会员", "积分", "权益", "车主", "绑定", "解绑", "换手机",
        "忘记密码", "验证码", "支付", "钱包", "充值",
    ],
    "navigation": [
        "导航", "地图", "路线", "OTA", "升级", "系统更新", "版本",
        "软件更新", "固件", "在线导航", "离线地图", "高德", "百度地图",
        "远程升级", "空中升级", "下载更新", "更新失败",
    ],
    "purchase": [
        "买", "购", "价格", "多少钱", "优惠", "分期", "贷款", "首付",
        "对比", "怎么选", "推荐", "预算", "落地价",
        "置换", "补贴", "金融", "订金", "定金",
        "哪个好", "值不值", "性价比", "选哪款",
    ],
    "vehicle_qa": [
        "续航", "电池", "加速", "几座", "颜色", "配置", "参数",
        "XNGP", "NGP", "ADS", "智驾", "泊车", "语音", "空间", "轴距",
        "功率", "扭矩", "风阻", "天幕", "悬架", "超充",
        "NFC", "钥匙", "规格", "尺寸", "重量",
        "换电", "哪款", "介绍", "区别", "差距", "怎么样", "评价", "好不好",
    ],
    "chitchat": [
        "你好", "谢谢", "再见", "你是谁", "笑话", "天气",
        "无聊", "哈哈", "厉害", "介绍一下你自己",
    ],
}

_ROUTE_TOOLS = [INTENT_TOOL_SCHEMA]


def _rule_classify(text: str) -> tuple[str, float]:
    scores: dict[str, float] = {k: 0.0 for k in _INTENT_KEYWORDS}
    for intent, kws in _INTENT_KEYWORDS.items():
        for kw in kws:
            if kw in text:
                scores[intent] += 1.0
    total = sum(scores.values())
    if total == 0:
        return "chitchat", 0.1
    best = max(scores, key=scores.__getitem__)
    return best, min(scores[best] / total, 0.99)


def _llm_classify(query: str, mem_ctx: str) -> tuple[str, float]:
    system = (
        "你是新能源汽车客服意图分类器，请调用 classify_intent 工具分类用户消息。\n"
        "12类意图：vehicle_qa(车型参数), aftersales(故障/保养), purchase(购车/金融), "
        "charging(充电问题), order_tracking(订单/交付), complaint(投诉/维权), "
        "account(账户/APP), insurance(保险/理赔), test_drive(试驾预约), "
        "navigation(导航/OTA), roadside(道路救援), chitchat(闲聊)"
    )
    if mem_ctx:
        system += f"\n用户历史背景：{mem_ctx}"
    result = llm_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": query}],
        max_tokens=64,
        temperature=0.0,
        tools=_ROUTE_TOOLS,
    )
    if isinstance(result, dict) and result.get("tool_calls"):
        tc = result["tool_calls"][0]
        args = tc.get("function", {}).get("arguments", tc.get("args", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        return args.get("intent", "chitchat"), args.get("confidence", 0.5)
    return "chitchat", 0.3


def router_node(state: CSState) -> dict:
    query = state["query"]
    session_id = state.get("session_id", "default")

    # 缓存命中 → 直接返回，后续节点全部跳过
    cached = cache_lookup(query)
    if cached:
        return {
            "intent": cached.get("intent", "vehicle_qa"),
            "entities": cached.get("entities", []),
            "answer": cached.get("answer", ""),
            "sources": cached.get("sources", []),
            "retrieved_chunks": [],
            "ticket_id": "",
            "step_count": 0,
            "cache_hit": True,
            "retrieval_trace": ["[cache] 语义缓存命中"],
            "memory_context": "",
            "backend": get_active_backend(),
        }

    # 读取用户历史偏好注入后续节点
    mem_ctx = memory_as_context(session_id)

    # 意图分类：规则 → LoRA → LLM
    intent, confidence = _rule_classify(query)
    if confidence < 0.6:
        try:
            from cs_agent.finetune.intent_classifier import classify_intent as lora_classify
            intent, confidence = lora_classify(query)
        except Exception:
            try:
                intent, confidence = _llm_classify(query, mem_ctx)
            except Exception:
                pass  # 保留规则结果

    # NER
    entities = extract_entities(query)

    # 写入本轮对话实体到长期记忆
    update_from_turn(session_id, query, intent, entities)

    return {
        "intent": intent,
        "intent_confidence": confidence,
        "entities": entities,
        "retrieved_chunks": [],
        "retrieval_trace": [],
        "ticket_id": "",
        "answer": "",
        "sources": [],
        "step_count": 0,
        "cache_hit": False,
        "memory_context": mem_ctx,
        "backend": get_active_backend(),
    }
