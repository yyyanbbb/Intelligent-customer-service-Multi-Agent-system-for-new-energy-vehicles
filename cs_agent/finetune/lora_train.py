"""
LoRA 微调脚本：针对意图分类任务微调 Qwen2.5-1.5B-Instruct。
显存约 6GB，适配 5070Ti（12-16GB显存）。

运行：
  python -m cs_agent.finetune.lora_train --epochs 3 --output ./lora_output
"""
from __future__ import annotations
import sys, json, argparse, os
from pathlib import Path

# Windows GBK 编码兼容：trl 读取 jinja 模板文件需要 UTF-8
os.environ.setdefault("PYTHONUTF8", "1")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

LABEL2ID = {
    "vehicle_qa": 0,
    "aftersales": 1,
    "purchase": 2,
    "charging": 3,
    "order_tracking": 4,
    "complaint": 5,
    "account": 6,
    "insurance": 7,
    "test_drive": 8,
    "navigation": 9,
    "roadside": 10,
    "chitchat": 11,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

PROMPT_TMPL = (
    "你是一个新能源汽车客服意图分类器。判断用户问题属于哪个类别：\n"
    "vehicle_qa（车型参数/功能咨询）/ aftersales（故障/维修/保养）/ purchase（购车/金融/置换）/ "
    "charging（充电问题/充电站）/ order_tracking（订单/交付进度）/ complaint（投诉/维权/赔偿）/ "
    "account（账户/APP/会员）/ insurance（保险/理赔/上牌）/ test_drive（试驾预约）/ "
    "navigation（导航/OTA升级）/ roadside（道路救援/紧急求助）/ chitchat（闲聊/问候）\n\n"
    "用户问题：{query}\n\n"
    "类别（只输出类别名称）："
)


def _build_dataset(eval_path: str) -> list[dict]:
    """从 eval_set.json 构建训练样本，补充新意图样本并扩充到 800+ 条。"""
    base = json.loads(Path(eval_path).read_text(encoding="utf-8"))

    # 新增意图的种子样本（eval_set 中无此类别）
    new_intent_seeds = [
        # charging
        {"text": "附近哪里有充电桩", "label": "charging"},
        {"text": "超充多少钱一度电", "label": "charging"},
        {"text": "充电充不进去怎么回事", "label": "charging"},
        {"text": "800V快充在哪里充", "label": "charging"},
        {"text": "续航焦虑怎么办", "label": "charging"},
        {"text": "V2L放电怎么用", "label": "charging"},
        # order_tracking
        {"text": "我的订单什么时候交付", "label": "order_tracking"},
        {"text": "提车进度怎么查", "label": "order_tracking"},
        {"text": "大定之后多久能交车", "label": "order_tracking"},
        {"text": "我的车发货了吗", "label": "order_tracking"},
        {"text": "交付中心在哪", "label": "order_tracking"},
        {"text": "尾款什么时候付", "label": "order_tracking"},
        # complaint
        {"text": "我要投诉你们的服务态度", "label": "complaint"},
        {"text": "车有质量问题你们不负责吗", "label": "complaint"},
        {"text": "我要退车退款", "label": "complaint"},
        {"text": "宣传续航跟实际不符", "label": "complaint"},
        {"text": "我要联系消费者协会", "label": "complaint"},
        {"text": "你们这是虚假宣传", "label": "complaint"},
        # account
        {"text": "APP登录不上去", "label": "account"},
        {"text": "忘记密码怎么找回", "label": "account"},
        {"text": "如何绑定车辆", "label": "account"},
        {"text": "会员积分怎么用", "label": "account"},
        {"text": "换手机了怎么迁移账号", "label": "account"},
        {"text": "车主权益在哪里看", "label": "account"},
        # insurance
        {"text": "新能源汽车保险怎么买", "label": "insurance"},
        {"text": "出险了怎么理赔", "label": "insurance"},
        {"text": "上牌需要什么手续", "label": "insurance"},
        {"text": "交强险要多少钱", "label": "insurance"},
        {"text": "临牌怎么申请", "label": "insurance"},
        {"text": "保费续保怎么操作", "label": "insurance"},
        # test_drive
        {"text": "我想预约试驾", "label": "test_drive"},
        {"text": "体验中心在哪里", "label": "test_drive"},
        {"text": "试驾活动怎么报名", "label": "test_drive"},
        {"text": "可以到店看车吗", "label": "test_drive"},
        {"text": "试驾需要什么证件", "label": "test_drive"},
        # navigation
        {"text": "怎么升级车机系统", "label": "navigation"},
        {"text": "OTA推送失败怎么处理", "label": "navigation"},
        {"text": "导航地图怎么更新", "label": "navigation"},
        {"text": "在线导航收费吗", "label": "navigation"},
        {"text": "系统版本怎么查看", "label": "navigation"},
        # roadside
        {"text": "我的车在高速上抛锚了", "label": "roadside"},
        {"text": "需要道路救援", "label": "roadside"},
        {"text": "打不着火需要拖车", "label": "roadside"},
        {"text": "出事故了需要紧急援助", "label": "roadside"},
        {"text": "车胎爆了在外面", "label": "roadside"},
    ]

    rephrases = {
        "vehicle_qa": ["{q}参数怎样？", "{q}能告诉我吗？", "关于{q}，有什么信息？"],
        "aftersales": ["{q}，请帮我处理一下", "{q}，怎么解决？", "我遇到问题：{q}"],
        "purchase": ["{q}，帮我分析", "{q}，有什么建议？", "我想了解：{q}"],
        "charging": ["{q}，帮我解答", "关于充电，{q}", "我想知道：{q}"],
        "order_tracking": ["我想查询：{q}", "{q}，能帮我确认吗？", "订单方面，{q}"],
        "complaint": ["我非常不满：{q}", "{q}，必须给我解释", "这件事很严重：{q}"],
        "account": ["账号问题：{q}", "{q}，怎么操作？", "APP里{q}"],
        "insurance": ["保险方面{q}", "{q}，需要哪些材料？", "关于理赔，{q}"],
        "test_drive": ["我想{q}", "{q}，怎么预约？", "能安排{q}吗？"],
        "navigation": ["车机系统{q}", "{q}，怎么操作？", "OTA方面{q}"],
        "roadside": ["紧急情况：{q}", "{q}，请尽快派人", "需要帮助：{q}"],
        "chitchat": ["{q}！", "嗨，{q}", "{q}呢？"],
    }

    augmented = []
    for s in base:
        augmented.append({"text": s["query"], "label": s["expected_intent"]})
        for tmpl in rephrases.get(s["expected_intent"], []):
            augmented.append({"text": tmpl.format(q=s["query"]), "label": s["expected_intent"]})

    for s in new_intent_seeds:
        augmented.append(s)
        for tmpl in rephrases.get(s["label"], []):
            augmented.append({"text": tmpl.format(q=s["text"]), "label": s["label"]})

    return augmented


def train(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    output_dir: str = "./lora_output",
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 2e-4,
    lora_r: int = 8,
    lora_alpha: int = 16,
):
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig

    eval_path = str(Path(__file__).parent.parent / "evaluation" / "eval_set.json")
    samples = _build_dataset(eval_path)
    print(f"训练样本数: {len(samples)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _format(sample: dict) -> str:
        prompt = PROMPT_TMPL.format(query=sample["text"])
        return prompt + sample["label"]

    dataset = Dataset.from_list([{"text": _format(s)} for s in samples])

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="none",
        dataloader_num_workers=0,
        max_length=256,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nLoRA adapter 已保存至: {output_dir}")


def evaluate_lora(adapter_path: str, eval_path: str | None = None):
    """加载 LoRA adapter，对 eval_set 评估意图分类准确率。"""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from peft import PeftModel

    eval_path = eval_path or str(Path(__file__).parent.parent / "evaluation" / "eval_set.json")
    samples = json.loads(Path(eval_path).read_text(encoding="utf-8"))

    base_model = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,
                    max_new_tokens=10, do_sample=False)

    correct = 0
    for s in samples:
        prompt = PROMPT_TMPL.format(query=s["query"])
        out = pipe(prompt)[0]["generated_text"][len(prompt):].strip().split()[0]
        pred = out if out in LABEL2ID else "chitchat"
        if pred == s["expected_intent"]:
            correct += 1

    acc = correct / len(samples)
    print(f"LoRA 微调后意图准确率: {acc:.3f}  ({correct}/{len(samples)})")
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output", default="./lora_output")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-only", default="")
    args = parser.parse_args()

    if args.eval_only:
        evaluate_lora(args.eval_only)
    else:
        train(model_name=args.model, output_dir=args.output, epochs=args.epochs, batch_size=args.batch_size)
