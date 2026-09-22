#!/usr/bin/env python3
"""Generate GUI element-matching training rows for jevy (Laya-style student). v3

v3 upgrades (from live jev-cua probing):
  1. goal PARAPHRASES: every goal has multiple phrasings (新建/添加/开一个 ...)
     so wording variation stops flipping the prediction to destructive buttons.
  2. multi-step menu trajectories: for menu-child goals, generate a CLOSED state
     (target = parent menu) and an OPEN state (children injected; target = the
     child item, parent becomes a distractor) — teaches "menu already open ->
     click the item, don't reopen or say done".
  3. destructive negatives: 关闭/最小化 buttons are force-included as
     distractors for benign goals, plus per-element noul rows
     ("元素X是否有助于目标G？") that teach the no-direction explicitly.

Rows mirror jev-cua `decide` exactly:
  state     = "目标: {goal}\n当前窗口: {title}\nidx{n} {role} {label} [{actions}]" ...
  action q  = choice over {click,type_text,press_key,hotkey,scroll,done}
  target q  = choice over {idx{n}: "{role} {label}"} (clickable only)
"""
import argparse
import hashlib
import json
import random

ACTIONS = ["click", "type_text", "press_key", "hotkey", "scroll", "done"]
YES, NO = "是", "否"

MENUS = {
    "文件": ["新建标签页", "新建窗口", "打开", "保存", "另存为", "打印"],
    "编辑": ["撤销", "剪切", "复制", "粘贴", "全选"],
    "设置及更多": ["设置", "历史记录", "扩展"],
}

# (window_title, [(role, label, [actions])...])
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

# goals: (paraphrases, target_label, action, menu_parent)
# menu_parent set -> also generates closed/open menu trajectories for windows
# that carry that menu, and the child item must exist in MENUS[menu_parent].
GOALS = [
    (["保存当前文件。", "保存文件。", "把文件存一下。", "点一下保存。", "保存。"], "保存", "click", "文件"),
    (["新建一个标签页。", "添加一个新标签页。", "再开一个标签页。", "新开个标签页。", "多开一个标签页来浏览。",
      "开个新标签页。", "我想新加一个标签页。"], "新建标签页", "click", "文件"),
    (["打开{app}的文件菜单。", "点开文件菜单。", "展开「文件」菜单。", "把文件菜单打开。"], "文件", "click", "文件"),
    (["关闭当前窗口。", "把这个窗口关掉。", "关闭窗口。"], "关闭", "click", None),
    (["把这个页面收藏起来。", "添加到收藏夹。", "收藏这个网页。"], "收藏夹", "click", None),
    (["回到上一页。", "返回上一页。", "后退一下。"], "后退", "click", None),
    (["新建一个文件夹。", "建个新文件夹。", "添加一个文件夹。"], "新建", "click", None),
    (["把这份文档另存为其他格式。", "另存为一份。", "将文件另存一下。"], "另存为", "click", "文件"),
    (["打开设置里的深色模式开关。", "切换深色模式。", "把深色模式打开。"], "深色模式", "click", None),
    (["检查系统有没有可用更新。", "检查一下更新。", "给系统检查更新。"], "检查更新", "click", None),
    (["给{contact}发一条消息后点发送。", "把消息发出去。", "点发送按钮。"], "发送", "click", None),
    (["登录这个账号。", "点登录按钮。", "登录。"], "登录", "click", None),
    (["运行当前脚本。", "把脚本跑起来。", "点运行。"], "运行", "click", None),
    (["在搜索框里输入{kw}。", "搜索一下{kw}。", "往搜索框输入{kw}。"], "搜索", "type_text", None),
    (["在文件名框里输入{kw}。", "文件名填{kw}。"], "文件名", "type_text", None),
    (["往输入区粘贴{kw}。", "在输入框里输入{kw}。", "输入{kw}。"], "输入", "type_text", None),
    (["在用户名栏填入{kw}。", "用户名输入{kw}。"], "用户名", "type_text", None),
    (["在地址栏输入{kw}然后回车。", "地址栏敲入{kw}。"], "搜索或输入 Web 地址", "type_text", None),
    (["用快捷键保存文件。", "按 Ctrl+S 保存。", "Ctrl+S 存盘。"], None, "hotkey", None),
    (["切换到上一个应用。", "用快捷键切应用。", "Alt+Tab 切换窗口。"], None, "hotkey", None),
    (["把弹出的菜单关掉。", "关掉这个弹出菜单。", "按 Esc 收起菜单。"], None, "press_key", None),
    (["按回车确认输入。", "回车确认。", "敲回车提交。"], None, "press_key", None),
    (["取消当前操作。", "取消。", "点了取消。"], None, "press_key", None),
    (["往下翻看更多内容。", "向下滚动页面。", "往下滑。"], None, "scroll", None),
    (["页面太长，向上滚动回到顶部。", "往上滚动。", "滚回顶部。"], None, "scroll", None),
    (["已经保存完了，任务结束。", "保存好了，不用再操作。", "完成了，收工。"], None, "done", None),
    (["目标已达成，不需要更多操作。", "搞定，结束任务。", "完成了，结束。"], None, "done", None),
]
KWS = ["deepseek", "报告2026", "todo清单", "meeting notes", "发票", "photo1"]
CONTACTS = ["文件传输助手", "张伟", "老板"]
DESTRUCTIVE = {"关闭", "最小化"}


def fill(tpl, rng):
    return (tpl.replace("{app}", rng.choice(["记事本", "浏览器", "资源管理器"]))
               .replace("{kw}", rng.choice(KWS))
               .replace("{contact}", rng.choice(CONTACTS)))


def label_match(label, target):
    if label == target:
        return 2
    if target in label:
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--windows", type=int, default=2500)
    ap.add_argument("--holdout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    action_rows, target_rows, noul_rows = [], [], []

    for wi in range(args.windows):
        title, elements = rng.choice(W)
        title = title if rng.random() < 0.85 else title + " (副本)"
        goal_tpl, target_label, action, menu = rng.choice(GOALS)
        goal = fill(rng.choice(goal_tpl) if isinstance(goal_tpl, list) else goal_tpl, rng)

        clickables = [e for e in elements if set(e[2]) & {"invoke", "expand", "select", "toggle", "press"}]

        # menu trajectory: (state_elements, target_element) variants
        variants = []  # (elements_variant, target_element_label, tag)
        if target_label is None:
            variants.append((elements, None, "direct"))
        else:
            direct = [e for e in clickables if label_match(e[1], target_label) == 2]
            if direct:
                variants.append((elements, target_label, "direct"))
        if menu and target_label in MENUS[menu]:
            parent = [e for e in clickables if e[1] == menu]
            if parent:
                children = [("MenuItem", c, ["invoke"]) for c in MENUS[menu]]
                # closed: menu children absent -> target is the parent menu
                without_children = [e for e in elements if e[1] not in MENUS[menu]]
                variants.append((without_children, menu, "menu_closed"))
                # open: children visible; flat same-label elements removed so the
                # target child is unambiguous (real UIA trees behave the same)
                open_elements = without_children + children
                variants.append((open_elements, target_label, "menu_open"))

        for vi, (var_elements, var_target, tag) in enumerate(variants):
            var_clickables = [e for e in var_elements
                              if set(e[2]) & {"invoke", "expand", "select", "toggle", "press"}]
            hit = ([e for e in var_clickables if label_match(e[1], var_target) == 2] or
                   [e for e in var_clickables if label_match(e[1], var_target) == 1]) \
                if var_target else []
            if not hit:
                continue
            # destructive distractors are always included when present
            forced = [e for e in var_clickables
                      if e[1] in DESTRUCTIVE and e not in hit]
            rest = [e for e in var_clickables if e not in hit and e not in forced]
            rng.shuffle(rest)
            k = rng.randint(3, 12)
            cand = hit + forced[:2] + rest[:k]
            rng.shuffle(cand)
            cand = cand[:rng.randint(4, 14)]

            idxs, state_lines = [], [f"目标: {goal}", f"当前窗口: {title}"]
            target_idx, tsoft, option_texts = None, {}, []
            for i, (role, label, acts) in enumerate(cand):
                idx = idxs[-1] + rng.randint(1, 4) if idxs else rng.randint(1, 8)
                idxs.append(idx)
                state_lines.append(f"idx{idx} {role} {label} [{','.join(acts)}]")
                key = f"idx{idx}"
                tsoft[key] = 0.02
                option_texts.append(f"{role} {label} ({key})")
                if label_match(label, var_target) == 2 and target_idx is None:
                    target_idx = idx
            if target_idx is None:
                continue
            state_text = "\n".join(state_lines)
            rid = hashlib.sha1(f"{wi}|{vi}|{goal}|{tag}".encode()).hexdigest()[:12]

            soft = {a: 0.02 for a in ACTIONS}
            soft[action] = round(1 - 0.02 * 5, 4)
            action_rows.append({
                "id": rid + "a", "message": state_text,
                "instructions": "为达成目标，下一步执行哪个动作",
                "intents": {a: a for a in ACTIONS},
                "teacher": {"intent": action, "dist": soft, "model": "programmatic"},
            })
            tsoft[f"idx{target_idx}"] = round(1 - 0.02 * (len(cand) - 1), 4)
            target_rows.append({
                "id": rid + "t", "message": state_text,
                "instructions": f"要达成目标「{goal}」，应操作哪个元素",
                "intents": tsoft, "option_texts": option_texts,
                "teacher": {"intent": f"idx{target_idx}",
                            "dist": {k: v for k, v in sorted(tsoft.items(), key=lambda kv: -kv[1])},
                            "model": "programmatic"},
            })

            # noul rows: 1 yes (target) + up to 2 no (destructive / random first)
            tkey = f"idx{target_idx}"
            ttext = f"{hit[0][0]} {hit[0][1]} ({tkey})" if hit else tkey
            noul_rows.append({
                "id": rid + "ny", "message": state_text,
                "instructions": f"元素「{ttext}」是否有助于完成目标「{goal}」？",
                "intents": {YES: "是", NO: "否"},
                "teacher": {"intent": YES, "dist": {YES: 0.9, NO: 0.1},
                            "model": "programmatic"},
            })
            no_pairs = [(f"idx{idxs[i]}", cand[i])
                        for i in range(len(cand)) if f"idx{idxs[i]}" != tkey]
            destructive_pairs = [p for p in no_pairs if p[1][1] in DESTRUCTIVE]
            rng.shuffle(no_pairs)
            picks = destructive_pairs[:1] + [p for p in no_pairs
                                             if p not in destructive_pairs[:1]][:1]
            for pick_key, pick_elem in picks:
                noul_rows.append({
                    "id": rid + "nn" + pick_key, "message": state_text,
                    "instructions": f"元素「{pick_elem[0]} {pick_elem[1]} ({pick_key})」是否有助于完成目标「{goal}」？",
                    "intents": {YES: "是", NO: "否"},
                    "teacher": {"intent": NO, "dist": {YES: 0.1, NO: 0.9},
                                "model": "programmatic"},
                })

    rng.shuffle(action_rows)
    rng.shuffle(target_rows)
    rng.shuffle(noul_rows)
    for name, rows in (("gui_action", action_rows), ("gui_target", target_rows),
                       ("gui_noul", noul_rows)):
        h = int(len(rows) * args.holdout)
        with open(f"{args.out_dir}/{name}.jsonl", "w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows[h:]) + "\n")
        with open(f"{args.out_dir}/{name}_test.jsonl", "w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows[:h]) + "\n")
        print(f"{name}: {len(rows) - h} train / {h} test")


if __name__ == "__main__":
    main()
