"""Copy the SFT checkpoints from the pod to saves/distill-<name> (step-96, step-192, step-288, epoch-1) as they appear.
Each adapter is copied once its size has stopped changing, then verified against the pod's sha256. Pod copies are NOT deleted
(the retail test needs them). Also keeps a local copy of the training log. Exits after epoch-1 is copied and verified.

Usage: python3 sync_distill_checkpoints.py [--interval 60]
Env: TEACHER_HOST (default root@154.54.102.23), TEACHER_PORT (default 14422), TEACHER_KEY.
"""
import hashlib, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SAVES = os.path.join(HERE, "..", "saves")
HOST = os.environ.get("TEACHER_HOST", "root@154.54.102.23")
PORT = os.environ.get("TEACHER_PORT", "14422")
KEY = os.path.expanduser(os.environ.get("TEACHER_KEY", "~/.ssh/id_ed25519"))
INTERVAL = int(sys.argv[sys.argv.index("--interval") + 1]) if "--interval" in sys.argv else 60


def ssh(cmd):
    r = subprocess.run(["ssh", "-o", "ConnectTimeout=20", "-p", PORT, "-i", KEY, HOST, cmd],
                       capture_output=True, text=True, stdin=subprocess.DEVNULL)
    return r.stdout.strip() if r.returncode == 0 else None


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def listing():
    out = ssh("for d in /root/sft_out/step-* /root/sft_out/epoch-*; do [ -f \"$d/adapter_model.safetensors\" ] && "
              "echo $(basename $d) $(stat -c %s \"$d/adapter_model.safetensors\"); done; true")
    if out is None:
        return None
    return {n: int(s) for n, s in (l.split() for l in out.splitlines() if l.strip())}


def main():
    last_sizes, done = {}, set()
    while True:
        cur = listing()
        if cur is not None:
            for name, size in cur.items():
                if name in done:
                    continue
                if last_sizes.get(name) != size:       # still being written (or just seen): wait for a stable size
                    last_sizes[name] = size
                    continue
                local = os.path.join(SAVES, f"distill-{name}")
                lf = os.path.join(local, "adapter_model.safetensors")
                if not os.path.exists(lf) or os.path.getsize(lf) != size:
                    os.makedirs(SAVES, exist_ok=True)
                    subprocess.run(["scp", "-q", "-r", "-P", PORT, "-i", KEY, f"{HOST}:/root/sft_out/{name}/.", local], check=False)
                if os.path.exists(lf) and os.path.getsize(lf) == size:
                    remote = (ssh(f"sha256sum /root/sft_out/{name}/adapter_model.safetensors") or "").split(" ")[0]
                    if remote and remote == sha(lf):
                        print(f"{name}: copied to saves/distill-{name} and verified", flush=True)
                        done.add(name)
                    else:
                        print(f"{name}: checksum mismatch, will retry", flush=True)
                        os.remove(lf)
            subprocess.run(["scp", "-q", "-P", PORT, "-i", KEY, f"{HOST}:/root/sft_out/train_log.jsonl",
                            os.path.join(SAVES, "distill-train_log.jsonl")], check=False, stderr=subprocess.DEVNULL)
            if "epoch-1" in done:
                print("final checkpoint saved locally; exiting", flush=True)
                break
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
