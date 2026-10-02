"""Which retail test tasks does the fine-tuned model solve that the starting adapter did not, and what kind of tasks are they?

Per task: new = share of its 2 trials passed, base = share passed by the starting adapter (the earlier test, plus the same-pod re-run
when teacher/data/baseline_rerun_results.jsonl exists). gained = new - base >= +0.5, lost <= -0.5.
Groups tasks (1) by scenario type from the ground-truth write actions, (2) by TF-IDF clusters of the task instructions (the "plots"),
(3) by scenario traits found in the instruction text. Writes full_run_results/new_solves_analysis.json and
renders/retail_test_new_solves.html.

Usage: ./.venv-nb/bin/python teacher/analyze_new_solves.py
"""
import collections, html, json, os, re

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TASKS = json.load(open(os.path.join(HERE, "data", "retail_test_tasks.json")))
NEW = os.path.join(HERE, "data", "distilled_retail_test_results.jsonl")
RERUN = os.path.join(HERE, "data", "baseline_rerun_results.jsonl")
EARLIER = os.path.join(ROOT, "full_run_results", "passk_run.json")
OUT_JSON = os.path.join(ROOT, "full_run_results", "new_solves_analysis.json")
OUT_HTML = os.path.join(ROOT, "renders", "retail_test_new_solves.html")

SHORT = {"cancel_pending_order": "cancel", "return_delivered_order_items": "return", "exchange_delivered_order_items": "exchange",
         "modify_pending_order_items": "modify items", "modify_pending_order_address": "change order address",
         "modify_pending_order_payment": "change payment", "modify_user_address": "change profile address"}
FEATURES = {
    "has a fallback / conditional ('if X is not possible, do Y')": r"\bif\b[^.]{0,160}\b(not|can't|cannot|unable|no\b)|otherwise|\binstead\b|but if",
    "customer changes their mind": r"change (your|his|her) mind|rethink|changes? (your|his|her) mind|actually",
    "withholds or does not know info": r"(do not|don't|doesn't|won't) (want to )?(reveal|remember|know|provide|mention)|not sure|forgot|can't remember|should be in",
    "difficult persona (angry, impatient, anxious, rude...)": r"angry|rude|impatient|anxious|upset|emotional|stress|swear|insecure|pessimistic",
    "asks for info (price, total, tracking, options)": r"tracking number|how much|total|price|options|what is the cheapest|wonder",
    "several orders involved": r"two orders|different orders|another order|both orders|all (the )?orders|each order",
}


def load_rows(path):
    return [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []


def per_task(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["task_index"]].append(1 if r["reward"] >= 1 else 0)
    return by


new = per_task(load_rows(NEW))
base_rows = json.load(open(EARLIER))["models"][1]["rows"] + load_rows(RERUN)
base = per_task(base_rows)
n_base_trials = len(base_rows) // len(TASKS)


def scenario(task):
    writes = [a["name"] for a in task["actions"] if a["name"] in SHORT]
    types = sorted({SHORT[w] for w in writes})
    if not types:
        coarse = "no write action (lookup or transfer)"
    elif len(types) == 1:
        coarse = types[0] + " only"
    else:
        coarse = "combo: 2+ action types"
    return coarse, types, len(writes)


rows = []
for i, t in enumerate(TASKS):
    coarse, types, nw = scenario(t)
    nr, br = sum(new[i]) / len(new[i]), sum(base[i]) / len(base[i])
    text = t["instruction"]
    rows.append({"task": i, "coarse": coarse, "types": types, "n_writes": nw, "new": nr, "base": br, "delta": nr - br,
                 "status": "gained" if nr - br >= 0.5 else ("lost" if nr - br <= -0.5 else "same"),
                 "newly_solved": nr == 1.0 and br == 0.0, "instruction": text, "actions": [a for a in t["actions"] if a["name"] in SHORT],
                 "new_trials": new[i], "base_trials": base[i],
                 "features": [k for k, rx in FEATURES.items() if re.search(rx, text, re.I)]})

# text clusters of the task stories
vec = TfidfVectorizer(stop_words="english", min_df=3, max_df=0.5, ngram_range=(1, 2))
X = vec.fit_transform([re.sub(r"[A-Z][a-z]+ [A-Z][a-z]+|\b\w+_\d{3,}\b|#W\d+|\b\d{5}\b", " ", r["instruction"]) for r in rows])
K = 6
km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(X)
terms = np.array(vec.get_feature_names_out())
clusters = []
for c in range(K):
    idx = [i for i, l in enumerate(km.labels_) if l == c]
    top = terms[np.argsort(km.cluster_centers_[c])[::-1][:6]].tolist()
    for i in idx:
        rows[i]["cluster"] = c
    clusters.append({"id": c, "top_terms": top, "tasks": idx})


def summarize(group):
    n = len(group)
    return {"tasks": n, "new_rate": sum(r["new"] for r in group) / n if n else None, "base_rate": sum(r["base"] for r in group) / n if n else None,
            "gained": sum(r["status"] == "gained" for r in group), "lost": sum(r["status"] == "lost" for r in group),
            "newly_solved": sum(r["newly_solved"] for r in group)}


by_coarse = {k: summarize([r for r in rows if r["coarse"] == k]) for k in sorted({r["coarse"] for r in rows})}
combo_pairs = collections.Counter(" + ".join(r["types"]) for r in rows if r["coarse"].startswith("combo"))
by_pair = {k: summarize([r for r in rows if r["coarse"].startswith("combo") and " + ".join(r["types"]) == k]) for k in combo_pairs}
by_feature = {k: summarize([r for r in rows if k in r["features"]]) for k in FEATURES}
by_feature["(none of these traits)"] = summarize([r for r in rows if not r["features"]])
by_cluster = {c["id"]: {**summarize([rows[i] for i in c["tasks"]]), "top_terms": c["top_terms"]} for c in clusters}
by_nwrites = {str(k): summarize([r for r in rows if min(r["n_writes"], 3) == k]) for k in (0, 1, 2, 3)}
overall = summarize(rows)

json.dump({"baseline_trials_per_task": n_base_trials, "overall": overall, "by_scenario": by_coarse, "by_combo_pair": by_pair,
           "by_cluster": by_cluster, "by_trait": by_feature, "by_number_of_write_actions": by_nwrites,
           "tasks": [{k: v for k, v in r.items() if k != "instruction"} for r in rows]}, open(OUT_JSON, "w"), indent=1)


# ---------- HTML report ----------
def esc(s):
    return "".join(f"&#{ord(c)};" if ord(c) > 127 else c for c in html.escape(str(s), quote=False))


def pct(x):
    return "-" if x is None else f"{x:.0%}"


def table(head, body):
    return ("<table><tr>" + "".join(f"<th>{esc(h)}</th>" for h in head) + "</tr>" +
            "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in body) + "</table>")


def srow(name, s):
    return [esc(name), s["tasks"], pct(s["base_rate"]), pct(s["new_rate"]), f"{s['new_rate'] - s['base_rate']:+.0%}" if s["tasks"] else "-",
            s["gained"], s["lost"], s["newly_solved"]]


HEAD = ["group", "tasks", "starting adapter", "new model", "change", "tasks gained", "tasks lost", "newly solved (0 to 2/2)"]


def task_card(r):
    acts = "".join(f"<div class='act'><b>{esc(a['name'])}</b> <code>{esc(json.dumps(a['kwargs'])[:260])}</code></div>" for a in r["actions"]) or "<div class='act'>(no write action)</div>"
    return (f"<details class='task'><summary><span class='idx'>#{r['task']}</span> <span class='tag'>{esc(r['coarse'])}</span> "
            f"<span class='res'>starting adapter {sum(r['base_trials'])}/{len(r['base_trials'])} &rarr; new {sum(r['new_trials'])}/{len(r['new_trials'])}</span> "
            f"{esc(r['instruction'][:150])}...</summary><div class='body'><p>{esc(r['instruction'])}</p>"
            f"<p class='lbl'>correct write actions</p>{acts}<p class='lbl'>traits: {esc(', '.join(r['features']) or 'none detected')} | "
            f"text cluster {r['cluster']}: {esc(', '.join(by_cluster[r['cluster']]['top_terms'][:4]))}</p></div></details>")


gained = sorted([r for r in rows if r["status"] == "gained"], key=lambda r: (r["coarse"], r["task"]))
lost = sorted([r for r in rows if r["status"] == "lost"], key=lambda r: (r["coarse"], r["task"]))
page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Retail test: what the fine-tuned model newly solves</title>
<style>:root{{color-scheme:light dark}}body{{font:15px/1.5 system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 18px}}
table{{border-collapse:collapse;margin:8px 0 22px;width:100%}}td,th{{border:1px solid #8885;padding:4px 9px;text-align:left}}th{{background:#8882}}
h2{{margin-top:34px}}.task{{border:1px solid #8885;border-radius:6px;margin:6px 0;padding:6px 10px}}.task summary{{cursor:pointer}}
.idx{{font-family:monospace;opacity:.7}}.tag{{background:#2F5D8A33;border-radius:4px;padding:1px 6px;font-size:.85em}}.res{{font-family:monospace;font-size:.85em;color:#1E7A5C}}
.lbl{{opacity:.7;margin:8px 0 2px;font-size:.9em}}.act{{margin:2px 0}}code{{font-size:.82em;overflow-wrap:anywhere}}.body p{{margin:6px 0}}</style></head><body>
<h1>Retail test: what the fine-tuned model newly solves</h1>
<p>Per task, the share of its 2 trials passed by the new model against the share passed by the starting adapter
({n_base_trials} baseline trials per task pooled). <b>gained</b> = up by at least half a task, <b>lost</b> = down by at least half.
115 tasks: {overall['gained']} gained, {overall['lost']} lost, {overall['tasks'] - overall['gained'] - overall['lost']} about the same;
{overall['newly_solved']} are newly solved outright (starting adapter never passed, new model passed both trials).
Each group below is small, so read differences of one or two tasks as noise.</p>
<h2>By scenario type (from the correct write actions)</h2>{table(HEAD, [srow(k, v) for k, v in by_coarse.items()])}
<h3>Combos by which actions are combined</h3>{table(HEAD, [srow(k, v) for k, v in sorted(by_pair.items(), key=lambda kv: -kv[1]['tasks'])])}
<h3>By number of write actions the task needs</h3>{table(HEAD, [srow(f"{k}{'+' if k == '3' else ''} write actions", v) for k, v in by_nwrites.items()])}
<h2>By story: clusters of the task instructions (TF-IDF, 6 clusters)</h2>
{table(HEAD, [srow(f"cluster {k}: " + ", ".join(v['top_terms'][:5]), v) for k, v in by_cluster.items()])}
<h2>By scenario trait found in the instruction text</h2>{table(HEAD, [srow(k, v) for k, v in by_feature.items()])}
<h2>Tasks the new model does better on ({len(gained)})</h2>{''.join(task_card(r) for r in gained)}
<h2>Tasks the new model does worse on ({len(lost)})</h2>{''.join(task_card(r) for r in lost)}
</body></html>"""
open(OUT_HTML, "w").write(page)

# ---------- console summary ----------
print(f"overall: {overall['tasks']} tasks | starting adapter {pct(overall['base_rate'])} -> new {pct(overall['new_rate'])} | gained {overall['gained']}, lost {overall['lost']}, newly solved {overall['newly_solved']}")
print("\nby scenario:")
for k, v in by_coarse.items():
    print(f"  {k:38} n={v['tasks']:>3}  {pct(v['base_rate']):>4} -> {pct(v['new_rate']):>4}  gained {v['gained']:>2}  lost {v['lost']:>2}  newly solved {v['newly_solved']}")
print("\nby text cluster:")
for k, v in by_cluster.items():
    print(f"  [{k}] {', '.join(v['top_terms'][:5]):52} n={v['tasks']:>3}  {pct(v['base_rate']):>4} -> {pct(v['new_rate']):>4}  gained {v['gained']:>2}  lost {v['lost']:>2}")
print("\nby trait:")
for k, v in by_feature.items():
    print(f"  {k[:58]:58} n={v['tasks']:>3}  {pct(v['base_rate']):>4} -> {pct(v['new_rate']):>4}  gained {v['gained']:>2}  lost {v['lost']:>2}")
print("\nreport:", os.path.relpath(OUT_HTML, ROOT))
