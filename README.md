# EV AI 新能源汽车智能客服（2026 版）

新能源汽车 Multi-Agent 客服系统，对标 2026 年主流 AI Agent 工程实践。

## 架构亮点

| 能力 | 实现 |
| --- | --- |
| 多节点路由 | LangGraph StateGraph + **12 类意图**分类（规则 → LoRA 1.5B → LLM tool-calling 三级回退） |
| 混合检索 | BM25 + FAISS dense（bge-small-zh）+ Reciprocal Rank Fusion |
| 重排 | BGE reranker-base cross-encoder |
| 反思检索 | Self-RAG：检索评分不足时 LLM 重写查询再检 |
| 结构化输出 | Pydantic schema（VehicleRecommendation / DiagnosisResult） |
| 长期记忆 | Mem0-lite：跨会话保存用户预算/偏好/关注车型 |
| 语义缓存 | bge 余弦相似度缓存，命中阈值 0.92，重复问题秒回 |
| 双 LLM 后端 | Ollama qwen3.5:4b（本地）+ DeepSeek API（云端），自动回退 |
| 流式输出 | Gradio 流式渲染 + Ollama/DeepSeek stream API |
| 可观测性 | 内存 trace ring-buffer，UI 侧边栏实时展示检索轨迹 |
| MCP Server | 6 个工具暴露为 stdio MCP，Claude Code/Cursor 可直接调用 |
| 知识库 | 181 条车型 + FAQ，支持一键重建 FAISS 索引 |
| LoRA 分类器 | Qwen2.5-1.5B-Instruct 微调，12 类意图，精度 100%，RTX 5070 Ti 推理 |

## 12 类意图路由

| 意图 | 说明 | 节点 |
| --- | --- | --- |
| `vehicle_qa` | 车型参数/功能咨询 | vehicle_qa（Hybrid RAG + Self-RAG） |
| `aftersales` | 故障/维修/保养 | aftersales（安全检查 + 工单） |
| `purchase` | 购车/金融/置换 | purchase（VehicleRecommendation） |
| `charging` | 充电问题/充电站 | charging（RAG） |
| `order_tracking` | 订单/交付进度 | order_tracking（RAG） |
| `complaint` | 投诉/维权/赔偿 | complaint（自动工单） |
| `roadside` | 道路救援/紧急求助 | roadside（紧急工单，最高优先级） |
| `account` | 账户/APP/会员 | chitchat（通用 LLM） |
| `insurance` | 保险/理赔/上牌 | chitchat（通用 LLM） |
| `test_drive` | 试驾预约 | chitchat（通用 LLM） |
| `navigation` | 导航/OTA 升级 | chitchat（通用 LLM） |
| `chitchat` | 闲聊/问候 | chitchat（通用 LLM） |

## 目录

```text
ev-ner-agent/
├─ cs_agent/
│  ├─ nodes/           # 9 个业务节点
│  │  ├─ router.py         # 意图分类（规则→LoRA→LLM 三级路由）
│  │  ├─ vehicle_qa.py     # 车型参数（Hybrid RAG + Self-RAG）
│  │  ├─ aftersales.py     # 售后故障（安全检查 + 工单）
│  │  ├─ purchase.py       # 购车咨询（结构化推荐）
│  │  ├─ charging.py       # 充电问题
│  │  ├─ order_tracking.py # 订单/交付
│  │  ├─ complaint.py      # 投诉处理（自动工单）
│  │  ├─ roadside.py       # 道路救援（紧急工单）
│  │  └─ chitchat.py       # 通用兜底
│  ├─ tools/           # ner_tool / rag_tool / hybrid_rag / ticket_tool
│  ├─ knowledge/       # vehicles.json / faq.json / memory/（长期记忆）
│  ├─ finetune/        # LoRA 意图分类器（lora_train.py + intent_classifier.py）
│  ├─ evaluation/      # 评估集 + 评估脚本
│  ├─ llm_client.py    # 双后端 LLM 层（Ollama + DeepSeek）
│  ├─ graph.py         # LangGraph 主图（含缓存短路 + Human-in-the-Loop）
│  ├─ state.py         # CSState TypedDict
│  ├─ schemas.py       # Pydantic 结构化 schema
│  ├─ memory.py        # 长期记忆 Mem0-lite
│  ├─ observability.py # 语义缓存 + trace
│  ├─ mcp_server.py    # MCP stdio server
│  └─ app.py           # Gradio UI
├─ ev_ner_agent/       # EV NER 子系统（PDF 电池参数提取，独立 ReAct 循环）
├─ scripts/            # 工具脚本（scrape_vehicles / visualize_graph 等）
├─ lora_output/        # LoRA adapter 输出（训练后生成）
├─ .mcp.json           # MCP server 配置
├─ .env                # 后端配置（不提交到 git）
├─ requirements.txt
└─ run.py              # 统一入口（cs + ner 两种模式）
```

## 快速开始

```bash
pip install -r requirements.txt

# 1. 启动本地 Ollama（需提前安装 + 拉取模型）
ollama serve
ollama pull qwen3.5:4b

# 2. 启动客服 UI（默认 http://localhost:7860）
# Windows 用户需加 PYTHONUTF8=1 避免编码问题
PYTHONUTF8=1 python run.py cs --ui

# 3. 切换 DeepSeek 后端
# 编辑 .env：LLM_BACKEND=deepseek，填入 DEEPSEEK_API_KEY
```

## 环境变量（.env）

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LLM_BACKEND` | `ollama` | 主后端：`ollama` 或 `deepseek` |
| `OLLAMA_MODEL` | `qwen3.5:4b` | Ollama 模型名 |
| `DEEPSEEK_API_KEY` | — | DeepSeek API 密钥（云端回退时使用） |
| `DEEPSEEK_MODEL` | `deepseek-chat` | DeepSeek 模型名 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | HuggingFace 镜像（中国大陆加速） |

## MCP Server

在支持 MCP 的客户端（Claude Code、Cursor）中，项目根目录已配置 `.mcp.json`。
可用工具：

| 工具 | 说明 |
| --- | --- |
| `ask_ev_agent` | 完整 multi-agent 对话 |
| `rag_search` | 混合 RAG 检索 |
| `extract_entities` | 车辆领域 NER |
| `create_ticket` | 生成售后工单 |
| `get_memory` | 查询会话记忆 |
| `list_recent_traces` | 查看调用 trace |

## LoRA 意图分类器

使用 Qwen2.5-1.5B-Instruct 微调，对 12 类意图进行轻量分类：

```bash
# 训练（约 66 秒，RTX 5070 Ti，300 样本，3 epoch）
PYTHONUTF8=1 HF_ENDPOINT=https://hf-mirror.com python -m cs_agent.finetune.lora_train

# 验证
python -c "from cs_agent.finetune.intent_classifier import classify_intent; print(classify_intent('刹车异响'))"
# 输出: ('aftersales', 0.92)
```

路由优先级：**规则关键词（0.99）→ LoRA 分类器（0.92）→ LLM tool-calling（回退）**

## 其他命令

```bash
PYTHONUTF8=1 python run.py cs --eval          # 运行评估集
PYTHONUTF8=1 python run.py cs --build-index   # 重建 FAISS 索引（知识库更新后）
python run.py ner --query "..."               # EV NER 子系统（电池参数提取）
```

## 已知限制

- **Windows + Node.js 24**：Gradio 6.x 在此环境下有 SSR 兼容问题，需使用 Gradio 5.50（requirements.txt 已锁定 `<6.0.0`）
- **Ollama 并发**：单线程 Ollama 服务在并发请求时可能返回 502，系统会自动回退到 DeepSeek API
- **LoRA 推理**：首次调用时需加载 Qwen2.5-1.5B 模型（约 3GB VRAM），之后懒加载缓存

## 技术栈

- **Multi-Agent 编排**：LangGraph StateGraph + 条件路由（缓存短路 + Human-in-the-Loop）
- **混合 RAG**：BM25 + FAISS dense + RRF 融合 + BGE reranker
- **Self-RAG 反思**：检索质量评分 + 查询重写再检
- **LoRA 意图分类**：Qwen2.5-1.5B + PEFT，三级路由（规则→LoRA→LLM）
- **长期记忆**：Mem0-lite（跨会话 JSON 存储 + 实体归因）
- **语义缓存**：bge 余弦相似度，重复问题 0 延迟
- **结构化输出**：Pydantic v2 schema + LLM tool-calling
- **双 LLM 后端**：Ollama（离线）+ DeepSeek API（云端）+ 自动回退
- **MCP Server**：6 工具 stdio 协议，Claude Code/Cursor 可调用
- **NER**：规则 + 正则，覆盖 180+ 车型 / 品牌 / 部件 / 故障 / 功能
