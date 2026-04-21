"""
LoRA 微调意图分类器推理封装。
懒加载：首次调用时才加载模型，避免影响主进程启动时间。
"""
from __future__ import annotations
from pathlib import Path

LABELS = [
    "vehicle_qa", "aftersales", "purchase", "charging",
    "order_tracking", "complaint", "account", "insurance",
    "test_drive", "navigation", "roadside", "chitchat",
]

# lora_output 在项目根目录
_ADAPTER_PATH = Path(__file__).parent.parent.parent / "lora_output"
_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

_model = None
_tokenizer = None
_pipe = None


def _load():
    global _model, _tokenizer, _pipe
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from peft import PeftModel

    if not _ADAPTER_PATH.exists():
        raise FileNotFoundError(
            f"LoRA adapter 未找到：{_ADAPTER_PATH}\n"
            "请先运行：python -m cs_agent.finetune.lora_train"
        )

    _tokenizer = AutoTokenizer.from_pretrained(str(_ADAPTER_PATH), trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        _BASE_MODEL,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    _model = PeftModel.from_pretrained(base, str(_ADAPTER_PATH))
    _model.eval()
    _pipe = pipeline(
        "text-generation",
        model=_model,
        tokenizer=_tokenizer,
        max_new_tokens=10,
        do_sample=False,
    )


_PROMPT_TMPL = (
    "你是一个新能源汽车客服意图分类器。判断用户问题属于哪个类别：\n"
    "vehicle_qa（车型参数/功能咨询）/ aftersales（故障/维修/保养）/ purchase（购车/金融/置换）/ "
    "charging（充电问题/充电站）/ order_tracking（订单/交付进度）/ complaint（投诉/维权/赔偿）/ "
    "account（账户/APP/会员）/ insurance（保险/理赔/上牌）/ test_drive（试驾预约）/ "
    "navigation（导航/OTA升级）/ roadside（道路救援/紧急求助）/ chitchat（闲聊/问候）\n\n"
    "用户问题：{query}\n\n"
    "类别（只输出类别名称）："
)


def classify_intent(query: str) -> tuple[str, float]:
    """返回 (intent_label, confidence)，confidence 固定为 0.92（微调模型高置信）。"""
    if _pipe is None:
        _load()
    prompt = _PROMPT_TMPL.format(query=query)
    out = _pipe(prompt)[0]["generated_text"][len(prompt):].strip().split()[0]
    label = next((l for l in LABELS if l in out), "chitchat")
    return label, 0.92
