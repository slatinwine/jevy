#!/usr/bin/env python3
"""Generate GUI element-matching training rows for jevy (Laya-style student).

Rows mirror EXACTLY what jev-cua's `decide` sends at inference:
  state     = "目标: {goal}\n当前窗口: {title}\nidx{n} {role} {label} [{actions}]" ...
  action q  = choice over {click,type_text,press_key,hotkey,scroll,done}
  target q  = choice over {idx{n}: "{role} {label}"} (clickable elements only)

Labels are programmatic ground truth (correct by construction), no API cost.
Split: --holdout 10% kept for eval.

Usage:
  python make_gui_data.py --out-dir data/
"""
import argparse
import hashlib
import json
import random

ACTIONS = ["click", "type_text", "press_key", "hotkey", "scroll", "done"]

# ---------------------------------------------------------------- windows
# (window_title, [(role, label, [actions])...]); C = clickable per cua rules
W = [
    ("无标题 - 记事本", [
        ("MenuItem", "文件", ["expand"]), ("MenuItem", "编辑", ["expand"]),
        ("MenuItem", "格式", ["expand"]), ("MenuItem", "查看", ["expand"]),
        ("Button", "关闭", ["press"]), ("Button", "最大化", ["press"]),
        ("Document", "文本编辑区", []),
        ("MenuItem", "新建标签页", ["invoke"]), ("MenuItem", "另存为", ["invoke"]),
    ]),
    ("deepseek.txt - 记事本", [
        ("MenuItem", "文件", ["expand"]), ("MenuItem", "编辑", ["expand"]),
        ("Button", "保存", ["invoke"]), ("Button", "关闭", ["press"]),
        ("Document", "文本编辑区", []),
        ("MenuItem", "打印", ["invoke"]),
    ]),
    ("新标签页 - Microsoft Edge", [
        ("TabItem", "新标签页", ["select"]), ("Button", "新建标签页", ["invoke"]),
        ("Edit", "搜索或输入 Web 地址", ["invoke"]), ("Button", "关闭", ["press"]),
        ("Button", "刷新", ["invoke"]), ("Button", "后退", ["invoke"]),
        ("MenuItem", "设置及更多", ["expand"]), ("Button", "收藏夹", ["invoke"]),
    ]),
    ("文件资源管理器", [
        ("Button", "后退", ["invoke"]), ("Button", "前进", ["invoke"]),
        ("Edit", "搜索", ["invoke"]), ("Button", "新建", ["expand"]),
        ("ListItem", "文档", ["select"]), ("ListItem", "下载", ["select"]),
        ("ListItem", "图片", ["select"]), ("Button", "关闭", ["press"]),
        ("MenuItem", "查看更多", ["expand"]),
    ]),
    ("另存为", [
        ("Edit", "文件名", ["invoke"]), ("Button", "保存", ["invoke"]),
        ("Button", "取消", ["invoke"]), ("ComboBox", "保存类型", ["expand"]),
        ("Button", "帮助", ["invoke"]),
    ]),
    ("计算器", [
        ("Button", "七", ["invoke"]), ("Button", "八", ["invoke"]),
        ("Button", "九", ["invoke"]), ("Button", "加", ["invoke"]),
        ("Button", "等于", ["invoke"]), ("Button", "清除", ["invoke"]),
        ("MenuItem", "导航", ["expand"]),
    ]),
    ("设置", [
        ("ListItem", "系统", ["select"]), ("ListItem", "蓝牙和其他设备", ["select"]),
        ("ListItem", "网络和 Internet", ["select"]), ("Button", "关闭", ["press"]),
        ("ToggleSwitch", "深色模式", ["toggle"]), ("Edit", "查找设置", ["invoke"]),
        ("Button", "检查更新", ["invoke"]),
    ]),
    ("Visual Studio Code", [
        ("TabItem", "train_intent.py", ["select"]), ("TabItem", "README.md", ["select"]),
        ("Button", "运行", ["invoke"]), ("Button", "调试", ["invoke"]),
        ("MenuItem", "终端", ["expand"]), ("MenuItem", "文件", ["expand"]),
        ("Button", "源代码管理", ["invoke"]),
    ]),
    ("微信", [
        ("Edit", "输入", ["invoke"]), ("Button", "发送", ["invoke"]),
        ("ListItem", "文件传输助手", ["select"]), ("Button", "聊天文件", ["invoke"]),
        ("Button", "朋友圈", ["invoke"]), ("MenuItem", "更多", ["expand"]),
    ]),
    ("登录 - 账户", [
        ("Edit", "用户名", ["invoke"]), ("Edit", "密码", ["invoke"]),
        ("Button", "登录", ["invoke"]), ("Button", "取消", ["invoke"]),
        ("CheckBox", "记住我", ["toggle"]), ("Hyperlink", "忘记密码", ["invoke"]),
    ]),
]

# ---------------------------------------------------------------- goals
# (goal_template, target_label_or_None, action, needs_window=prefix filter)
G = [
    ("保存当前文件。", "保存", "click"),
    ("保存文件。", "保存", "click"),
    ("点击保存按钮。", "保存", "click"),
    ("新建一个标签页。", "新建标签页", "click"),
    ("打开{app}的文件菜单。", "文件", "click"),
    ("关闭当前窗口。", "关闭", "click"),
    ("把这个页面收藏起来。", "收藏夹", "click"),
    ("回到上一页。", "后退", "click"),
    ("新建一个文件夹。", "新建", "click"),
    ("把这份文档另存为其他格式。", "另存为", "click"),
    ("打开设置里的深色模式开关。", "深色模式", "click"),
    ("检查系统有没有可用更新。", "检查更新", "click"),
    ("给{contact}发一条消息后点发送。", "发送", "click"),
    ("登录这个账号。", "登录", "click"),
    ("运行当前脚本。", "运行", "click"),
    ("把这张图片收藏到下载列表。", "下载", "click"),
    ("在搜索框里输入{kw}。", "搜索", "type_text"),
    ("在文件名框里输入{kw}。", "文件名", "type_text"),
    ("往输入区粘贴{kw}。", "输入", "type_text"),
    ("在用户名栏填入{kw}。", "用户名", "type_text"),
    ("在地址栏输入{kw}然后回车。", "搜索或输入 Web 地址", "type_text"),
    ("用快捷键保存文件。", None, "hotkey"),
    ("按 Ctrl+S 保存。", None, "hotkey"),
    ("切换到上一个应用。", None, "hotkey"),
    ("把弹出的菜单关掉。", None, "press_key"),
    ("按回车确认输入。", None, "press_key"),
    ("取消当前操作。", None, "press_key"),
    ("往下翻看更多内容。", None, "scroll"),
  ("页面太长，向上滚动回到顶部。", None, "scroll"),
    ("已经保存完了，任务结束。", None, "done"),
    ("目标已达成，不需要更多操作。", None, "done"),
]
KWS = ["deepseek", "报告2026", "todo清单", "meeting notes", "发票", "photo1"]
CONTACTS = ["文件传输助手", "张伟", "老板"]


def fill(tpl, rng):
    return (tpl.replace("{app}", rng.choice(["记事本", "浏览器", "资源管理器"]))
               .replace("{kw}", rng.choice(KWS))
               .replace("{contact}", rng.choice(CONTACTS)))


def match(elem_label, goal_target):
    """Exact label match wins; substring only as fallback (avoids 文件 vs 文件传输助手)."""
    if elem_label == goal_target:
        return 2
    if goal_target in elem_label:
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--windows", type=int, default=700)
    ap.add_argument("--holdout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    action_rows, target_rows = [], []
    for wi in range(args.windows):
        title, elements = rng.choice(W)
        title = title if rng.random() < 0.8 else title + " (复制)"
        goal_tpl, target_label, action = rng.choice(G)
        goal = fill(goal_tpl, rng)
        if target_label and target_label == "文件" and rng.random() < 0.5:
            target_label = "文件" if "文件" in goal_tpl else target_label

        clickables = [e for e in elements if set(e[2]) & {"invoke", "expand", "select", "toggle", "press"}]
        if target_label:
            hits = [e for e in clickables if match(e[1], target_label) == 2] or \
                   [e for e in clickables if match(e[1], target_label) == 1]
            if hits:
                clickables = hits + [e for e in clickables if e not in hits]
        rng.shuffle(clickables)
        clickables = clickables[:rng.randint(4, min(15, len(clickables) + 3))] if clickables else []

        # indices: target gets a random distinct index
        idx_pool = rng.sample(range(1, 60), k=len(clickables) + 1)
        state_lines = [f"目标: {goal}", f"当前窗口: {title}"]
        idxs, target_idx = [], None
        for i, (role, label, acts) in enumerate(clickables):
            idx = idx_pool[i]
            state_lines.append(f"idx{idx} {role} {label} [{','.join(acts)}]")
            idxs.append(idx)
            if target_label and match(label, target_label) and target_idx is None:
                target_idx = idx
        if target_label and target_idx is None:
            continue  # goal does not apply to this window
        state_text = "\n".join(state_lines)

        # action row
        soft = {a: round(0.02, 4) for a in ACTIONS}
        soft[action] = round(1 - 0.02 * 5, 4)
        action_rows.append({
            "id": hashlib.sha1(f"ga|{wi}|{goal}|{action}".encode()).hexdigest()[:12],
            "message": state_text,
            "instructions": "为达成目标，下一步执行哪个动作",
            "intents": {a: a for a in ACTIONS},
            "teacher": {"intent": action, "dist": soft, "model": "programmatic"},
        })

        # target row (only when the goal names an element present in the tree)
        if target_label and target_idx is not None and len(clickables) >= 2:
            tsoft, option_texts = {}, []
            for i, (role, label, acts) in enumerate(clickables):
                key = f"idx{idxs[i]}"
                tsoft[key] = 0.02
                # option TEXT carries the semantic label (goal<->label matching);
                # the (idxN) suffix keeps duplicate labels unique and matches the
                # f"{criteria[k]} ({k})" texts the server derives at inference
                option_texts.append(f"{role} {label} ({key})")
            tsoft[f"idx{target_idx}"] = round(1 - 0.02 * (len(clickables) - 1), 4)
            target_rows.append({
                "id": hashlib.sha1(f"gt|{wi}|{goal}".encode()).hexdigest()[:12],
                "message": state_text,
                "instructions": f"要达成目标「{goal}」，应操作哪个元素",
                "intents": tsoft,
                "option_texts": option_texts,
                "teacher": {"intent": f"idx{target_idx}",
                            "dist": {k: v for k, v in sorted(tsoft.items(), key=lambda kv: -kv[1])},
                            "model": "programmatic"},
            })

    rng.shuffle(action_rows)
    rng.shuffle(target_rows)
    n_hold_a = int(len(action_rows) * args.holdout)
    n_hold_t = int(len(target_rows) * args.holdout)
    for name, rows, hold_n in (("gui_action", action_rows, n_hold_a), ("gui_target", target_rows, n_hold_t)):
        with open(f"{args.out_dir}/{name}.jsonl", "w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows[hold_n:]) + "\n")
        with open(f"{args.out_dir}/{name}_test.jsonl", "w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows[:hold_n]) + "\n")
        print(f"{name}: {len(rows) - hold_n} train / {hold_n} test")


if __name__ == "__main__":
    main()
