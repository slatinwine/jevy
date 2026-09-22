# jevy — 自炼的 Jev 式类型化判断模型

仿照 [Laya](https://huggingface.co/convaiinnovations/laya-typed-decisions) 的思路，
在一张 **GTX 1650（4GB 显存）** 上微调出的 **Jev 式类型化决策模型**（118M 参数，中英文）。

> [Jev](https://typesafe.ai) 是 TypeSafe AI 的 "System One" 模型：输入状态 + 类型化问题，
> 输出**带概率的决策**而非生成文本。jevy 是它的开源平替之一——用官方 Jev API 当老师
> 蒸馏而来，专做**意图识别 / 候选择优**这类判断任务。

- **单次前向**输出所有候选的概率分布，零自回归解码
- **候选数量推理时随意变**——加新意图不用重训
- **4GB 显存可训**：全量重训约 10 分钟；数据、训练脚本、评测集全部开源

## 目录

- [对比结果](#对比结果)
- [快速开始](#快速开始)
- [API](#api)
- [自己训一个](#自己训一个)
- [训练数据](#训练数据)
- [实现细节](#实现细节)
- [已知边界](#已知边界)

## 对比结果

所有数字均为实测（硬件：GTX 1650 4GB；老师：官方 `jev-1.13.0`）。

### 1) 与官方 Jev 的判断一致率（合成意图域，6 条 held-out 消息）

| 模型 | 一致率 | 备注 |
|---|---|---|
| NanoJev（0.6B，游戏域） | 33% | 领域不匹配 |
| Laya（421M，客服域） | 83% | 唯一分歧：发票导出判 billing（老师判 other） |
| **jevy（118M，本仓库）** | **100%** | 训练数据 771 条蒸馏样本 |

### 2) jevy 内部两轮迭代（dev 一致率，混合域）

| 版本 | 蒸馏数据 | dev 一致率 | held-out |
|---|---|---|---|
| v1 | 271 条模板消息 | 87%（54 条 dev） | 5/6 |
| v2 | 771 条（+500 模板、难例过采样） | **92%（154 条 dev）** | **6/6** |

### 3) 真实工单域（公开数据集真实工单，10 类意图，40 条从未参与训练）

| 版本 | 真实标注量 | 与官方一致率 |
|---|---|---|
| v0.1 | 260 条 | 35%（随机基线 10%） |
| **v0.2** | **1000 条** | **60%** |

真实文本多样性远超模板——继续扩充真实标注量仍是最有效的提升路径
（源数据集还有 1.5 万条可标）。

### 3.5) 三种题型（v0.2 起支持 choice / noul(布尔) / score(等级)）

| 题型 | 评测 | 结果 |
|---|---|---|
| choice | 真实工单 40 条 vs 官方 | 60% |
| noul（布尔） | 80 问（正例 vs 干扰项） | **84%** |
| score（紧急度 3 级） | 100 条 held-out | MAE 0.83 级（偏弱：标签源自数据集 priority 字段，噪声较大） |

### 3.6) GUI 元素匹配域（v0.3 新增）

配合 [cua-driver](https://github.com/trycua/cua) 电脑控制使用：给定 UIA 元素树文本
（`目标 / 窗口标题 / idx{n} 角色 标签 [动作]` 行）+ 一句自然语言目标，
模型输出 1) 下一步**动作类型**（click/type_text/press_key/hotkey/scroll/done），
2) 应操作的**目标元素**。数据由 `make_gui_data.py` 程序化合成（零 API 成本），
state/问题格式与 jev-cua 的 `decide` 请求逐字段对齐。

| 评测（held-out） | 结果 | 训练前 |
|---|---|---|
| 动作类型选择（127 条） | **100%** | ~17%（6 类随机） |
| 目标元素选择（31 条） | 26% | ~10% |

目标元素仍是当前短板——扩充窗口/目标模板（`make_gui_data.py` 的 `W`/`G` 表）
是最直接的提升方式。同时注意：多域混合训练需关注灾难性遗忘，
每个域的 held-out 都要在每次训练后回归（本仓库 data/ 附带全部测试集）。

### 4) 常识陷阱题：*"洗车店离这里 50 米，开车去还是走路去洗车？"*

正确答案：**开车**——要去洗的是车。

| 模型 | 判断 | 分布 |
|---|---|---|
| Laya | ✅ 开车 | 66% / 34% |
| jevy | ✅ 开车 | 63% / 37% |
| 官方 jev-1.13.0（英文转译输入） | ⚠️ 走路 | walk 83% / drive 41% |

### 5) 实测发现：官方 API 拒绝中文输入

`api.typesafe.ai` 的网关对非 ASCII 内容直接返回 403（key 有效性与此无关，
同请求换英文即 200）。中文场景请翻译后调用官方，或直接用本地后端。

## 快速开始

```bash
pip install -r requirements.txt
```

### 方式 A：下载训练好的权重（GitHub Releases）

从 [Releases](../../releases) 下载 `jevy-model.zip`，解压为 `model/`，然后：

```bash
python intent_server.py --model-dir model --port 8767
```

### 方式 B：自己训（约 10 分钟，4GB 显存即可）

训练数据已包含在本仓库 `data/`，直接复现：

```bash
python train_intent.py \
    --data data/train_synthetic_v2.jsonl data/train_synthetic_v3.jsonl \
           data/train_synthetic_v4.jsonl data/train_real.jsonl \
    --out model
python intent_server.py --model-dir model --port 8767
```

## API

与 NanoJev / Laya 服务器同契约（`POST /api/v1/decide`）：

```bash
curl -X POST http://127.0.0.1:8767/api/v1/decide \
  -H "Content-Type: application/json" \
  -d "{\"state\": \"我的手机充不进电，买了才两周。\", \
       \"questions\": {\"intent\": {\"type\": \"choice\", \
         \"instructions\": \"判断用户消息最符合哪个意图\", \
         \"criteria\": {\"hardware\": \"硬件故障\", \"billing\": \"账单问题\", \"other\": \"其他\"}}}}"
```

返回：

```json
{"object": "decision", "model": "jev-distill-student",
 "decisions": {"intent": {"type": "choice", "value": "hardware",
   "probabilities": {"hardware": 0.55, "other": 0.15, "billing": 0.02, ...},
   "confidence": 0.55}}}
```

也支持批量（`{"states": [{"id", "state", "questions"}, ...]}`，单请求最多 128 条）。
Python 调用见 [`examples/intent.py`](examples/intent.py)。

注意：候选 `criteria` 的数量与内容**推理时随意指定**，无需重训——
这是标记位架构的直接好处。

布尔（noul）与等级（score）问题：

```bash
# 布尔：返回 value/p_true
curl -X POST http://127.0.0.1:8767/api/v1/decide -H "Content-Type: application/json" \
  -d "{\"state\": \"登录一直转圈进不去。\", \
       \"questions\": {\"is_technical\": {\"type\": \"bool\", \
         \"instructions\": \"这条消息是否属于技术故障？\"}}}"

# 等级：criteria 为有序等级数组，返回分布 + expected（1 起始的期望等级）
curl -X POST http://127.0.0.1:8767/api/v1/decide -H "Content-Type: application/json" \
  -d "{\"state\": \"服务器宕机两小时了，客户全部投诉！\", \
       \"questions\": {\"urgency\": {\"type\": \"score\", \
         \"instructions\": \"评估这条消息的紧急程度。\", \
         \"criteria\": [\"low\", \"medium\", \"high\"]}}}"
```

score 类型当前仅用紧急度 3 级数据训练，其他等级语义需要补充对应训练行。

## 自己训一个

三步，全部脚本化：

```bash
# 1. 写你的意图集和消息清单（UTF-8，每行一条）
#    data/intents_synthetic.json: {"weather": "查天气", "music": "播放音乐", ...}

# 2. 官方 Jev 当老师打软标签（需要 TYPESAFE_API_KEY；中文需先转英文，见上文发现）
python collect_distill.py --messages-file msgs.txt \
    --intents-file data/intents_synthetic.json \
    --api-key $TYPESAFE_API_KEY --out data/train_mine.jsonl

# 3.（可选）从标注记录派生 noul/score 训练行，让一个模型支持三种题型
python make_typed_data.py --labeled data/train_mine.jsonl \
    --tickets-meta real_tickets_meta.jsonl --out-dir data/

# 4. 蒸馏训练 + 温度校准，然后起服务
python train_intent.py --data data/train_mine.jsonl data/typed_noul.jsonl data/typed_score_train.jsonl --out model
python intent_server.py --model-dir model --port 8767
```

数据格式（软标签蒸馏，一行一条）：

```json
{"id": "a11b2dcb", "message": "Where is my order #12345?",
 "intents": {"shipping": "物流问题", "...": "..."},
 "teacher": {"intent": "shipping",
             "dist": {"shipping": 0.71, "other": 0.12, "...": "..."},
             "model": "jev-1.13.0 (official)"}}
```

## 实现细节

Laya 的三个核心思路的迷你实现：

1. **编码器而非解码器**：`paraphrase-multilingual-MiniLM-L12-v2`（118M，XLM-R 系，中英通吃），
   一次前向给所有候选打分；
2. **选项标记位**：每个候选前插一个 `[OPT]` 标记 token，线性头读标记位隐状态出 logit——
   因此候选数量动态可变；
3. **软标签蒸馏 + 温度校准**：老师 noul 分布做 KL 蒸馏（比硬标签信号多），
   训后在 dev 上拟合温度修概率过冲；另带 best-dev 选点与难例过采样
   （学生与老师分歧 ×3 的样本）。

训练配置：batch 16、AdamW（头 1e-3 / 底座 3e-5）、fp16 autocast、约 1600 步，
GTX 1650 上 10~15 分钟。

## 已知边界

- 真实工单域 v0.2 为 60%（见对比结果 3），继续扩充真实标注仍是首推路径
- 底座只测到 118M：278M mpnet + Adafactor 在 4GB 卡上 fp16 数值不稳（dev 掉到 47% 已回退），
  更大底座建议冻结底层或用 8-bit 优化器
- `score` 类型仅用紧急度 3 级（low/medium/high）训练，其他等级语义需要补训练行；
  训练行的 `instructions` 支持每条自定义（trainer 自动读取）
- 官方 API 计费按 token，标注 1000 条约 40 万 input token——注意配额

## 数据与许可

- 代码：MIT
- `data/train_real.jsonl` / `data/test_real.jsonl` 来自公开数据集
  [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)
  （CC BY-NC 4.0），**仅限非商业用途**；合成消息与老师标注为本项目生成
- 感谢 TypeSafe AI 的 Jev 与开源平替社区（[awesome-jev](https://github.com/yibie/awesome-jev)、
  NanoJev、Laya）提供的思路与对照
