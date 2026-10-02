"""Copy the 32B teacher's rollouts from the pod and save the PASSED ones as SFT-ready JSON.

1. scp /root/teacher_rollouts.jsonl from the pod to teacher/data/teacher_rollouts_raw.jsonl (a full local copy of every
   rollout, passed or not, so the conversations survive the pod).
2. Keep rollouts with reward 1 (not skipped, no error) whose conversation is structurally valid.
3. Clean each conversation the way the 14B's SFT code did: only role/content/tool_calls/tool_call_id/name, content None -> "",
   no provider fields, no reasoning (add --keep-reasoning to keep the model's thinking as `reasoning_content`).
4. Write
     teacher/data/teacher32b_passed_sft[_reasoning].json    list of {"messages": [...], "tools": [...]}   (same schema as
                                                 dataset/qwen3_14b_retail_train_rollout_sft.json)
     teacher/data/teacher32b_passed_meta.json   aligned list: task_index, trial, 14B group, turn counts

Usage: python3 export_sft.py [--no-pull] [--keep-reasoning] [--groups never,mixed|all]   (default: only the hard groups)
Env: TEACHER_HOST (default root@154.54.102.23), TEACHER_PORT (default 14422), TEACHER_KEY.
"""
import collections, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RAW = os.path.join(DATA, "teacher_rollouts_raw.jsonl")
SUFFIX = "_reasoning" if "--keep-reasoning" in sys.argv else ""   # reasoning-kept export goes to separate files
OUT_SFT = None   # set after the paths are final (see main)
OUT_META = None
TOOLS_SRC = os.path.join(HERE, "..", "github_repo", "dataset", "qwen3_14b_retail_train_rollout_sft.json")
GROUPS = os.path.join(HERE, "..", "full_run_results", "combo_passrate.json")
HOST = os.environ.get("TEACHER_HOST", "root@154.54.102.23")
PORT = os.environ.get("TEACHER_PORT", "14422")
KEY = os.path.expanduser(os.environ.get("TEACHER_KEY", "~/.ssh/id_ed25519"))
ON_POD = os.environ.get("TEACHER_ON_POD") == "1"   # run on the pod itself: read /root/*, no copying
if ON_POD:
    RAW, DATA = "/root/teacher_rollouts.jsonl", "/root"
    TOOLS_SRC, GROUPS = "/root/tools.json", "/root/groups.json"
KEEP_REASONING = "--keep-reasoning" in sys.argv
GROUPS_KEEP = (sys.argv[sys.argv.index("--groups") + 1] if "--groups" in sys.argv else "never,mixed").split(",")  # "all" keeps every group


def pull():
    os.makedirs(DATA, exist_ok=True)
    tmp = RAW + ".part"
    r = subprocess.run(["scp", "-q", "-o", "ConnectTimeout=20", "-P", PORT, "-i", KEY,
                        f"{HOST}:/root/teacher_rollouts.jsonl", tmp], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"copy from the pod failed: {r.stderr.strip()[:200]}")
    os.replace(tmp, RAW)
    print(f"copied the pod's rollouts to {os.path.relpath(RAW, HERE)} ({os.path.getsize(RAW) / 1e6:.1f} MB)")


def clean_message(m):
    out = {"role": m["role"], "content": m.get("content") if m.get("content") is not None else ""}
    if m.get("tool_calls"):
        out["tool_calls"] = m["tool_calls"]
    if m["role"] == "tool":
        out["tool_call_id"] = m.get("tool_call_id")
        out["name"] = m.get("name")
    if KEEP_REASONING and m.get("reasoning_content"):
        out["reasoning_content"] = m["reasoning_content"]
    return out


def valid(msgs):
    """Structure the chat template and the loss masking rely on."""
    # tau-bench conversations end with the simulated customer's "###STOP###" (418 of the 422 existing SFT examples do too);
    # training only learns from the assistant turns, so the last message may be the customer's.
    if not msgs or msgs[0]["role"] != "system" or msgs[-1]["role"] == "tool":
        return False
    if not any(m["role"] == "assistant" for m in msgs):
        return False
    open_ids = set()
    for m in msgs:
        if m["role"] == "assistant":
            open_ids = {tc.get("id") for tc in (m.get("tool_calls") or [])}
        elif m["role"] == "tool":
            if m.get("tool_call_id") not in open_ids:
                return False
            open_ids.discard(m.get("tool_call_id"))
        elif m["role"] not in ("system", "user"):
            return False
    return True


def main():
    global OUT_SFT, OUT_META
    OUT_SFT = os.path.join(DATA, f"teacher32b_passed_sft{SUFFIX}.json")
    OUT_META = os.path.join(DATA, f"teacher32b_passed_meta{SUFFIX}.json")
    if "--no-pull" not in sys.argv and not ON_POD:
        pull()
    if ON_POD:
        tools = json.load(open(TOOLS_SRC))
        cat = {int(k): v for k, v in json.load(open(GROUPS)).items()}
    else:
        tools = json.load(open(TOOLS_SRC))[0]["tools"]
        cat = {p["task_index"]: p["category"].split(" ")[0] for p in json.load(open(GROUPS))["per_task"]}
    sft, meta, stats = [], [], collections.Counter()
    for line in open(RAW):
        if not line.strip():
            continue
        r = json.loads(line)
        stats["rollouts"] += 1
        if r.get("skipped"):
            stats["skipped"] += 1
        elif r.get("error"):
            stats["errored"] += 1
        elif (r.get("reward") or 0) < 1:
            stats["failed"] += 1
        else:
            msgs = [clean_message(m) for m in r["messages"]]
            if not valid(msgs):
                stats["passed_but_invalid_structure"] += 1
                continue
            if "all" not in GROUPS_KEEP and cat.get(r["task_index"], "?") not in GROUPS_KEEP:
                stats["passed_but_easy_group_skipped"] += 1
                continue
            stats["passed"] += 1
            sft.append({"messages": msgs, "tools": tools})
            meta.append({"task_index": r["task_index"], "trial": r["trial"], "group_14b": cat.get(r["task_index"], "?"),
                         "n_messages": len(msgs), "n_assistant_turns": sum(1 for m in msgs if m["role"] == "assistant"),
                         "n_tool_calls": sum(len(m.get("tool_calls") or []) for m in msgs),
                         "write_calls": sum(1 for m in msgs for tc in (m.get("tool_calls") or [])
                                            if tc["function"]["name"] not in ("find_user_id_by_email", "find_user_id_by_name_zip",
                                                                              "get_order_details", "get_product_details",
                                                                              "get_user_details", "list_all_product_types",
                                                                              "calculate", "think", "transfer_to_human_agents"))})
    tmp = OUT_SFT + ".tmp"
    json.dump(sft, open(tmp, "w"), indent=1, ensure_ascii=False)
    os.replace(tmp, OUT_SFT)
    json.dump(meta, open(OUT_META, "w"), indent=1)
    tasks = {m["task_index"] for m in meta}
    by_group = collections.Counter(m["group_14b"] for m in meta)
    print(f"rollouts read: {stats['rollouts']} | passed: {stats['passed']} | failed: {stats['failed']} | "
          f"errored: {stats['errored']} | skipped: {stats['skipped']} | passed but invalid structure: {stats['passed_but_invalid_structure']} | "
          f"passed in easy groups, left out: {stats['passed_but_easy_group_skipped']}")
    print(f"saved {len(sft)} SFT conversations from {len(tasks)} solved tasks -> {os.path.relpath(OUT_SFT, HERE)}")
    print("conversations by the 14B's group on that task:", dict(by_group))
    if meta:
        print(f"avg assistant turns {sum(m['n_assistant_turns'] for m in meta) / len(meta):.1f}, "
              f"avg write actions {sum(m['write_calls'] for m in meta) / len(meta):.1f}")


if __name__ == "__main__":
    main()
