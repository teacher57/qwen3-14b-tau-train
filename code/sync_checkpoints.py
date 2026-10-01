"""Copy probed GRPO checkpoints from the pod to saves/combo-round-N, verify sha256, then
delete the pod copy (the pod disk is nearly full). Never deletes unless the checksums match.
Usage: python3 sync_checkpoints.py [--interval 60]   (exits when the trainer has stopped and nothing is left)
"""
import hashlib, os, subprocess, sys, time

HOST, PORT = "root@194.68.245.68", "22050"
KEY = os.path.expanduser("~/.ssh/id_ed25519")
SAVES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saves")
INTERVAL = int(sys.argv[sys.argv.index("--interval") + 1]) if "--interval" in sys.argv else 60
PREFIX = sys.argv[sys.argv.index("--prefix") + 1] if "--prefix" in sys.argv else "combo-round"


def ssh(cmd):
    r = subprocess.run(["ssh", "-o", "ConnectTimeout=15", "-p", PORT, "-i", KEY, HOST, cmd],
                       capture_output=True, text=True, stdin=subprocess.DEVNULL)
    return r.stdout.strip() if r.returncode == 0 else None


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sync_once():
    # every checkpoint dir on the pod: "<n> <probe_done 0/1> <size of adapter file>"
    listing = ssh("for d in /root/grpo_checkpoints/round-*; do [ -d \"$d\" ] || continue; n=${d##*round-}; "
                  "f=$d/adapter_model.safetensors; [ -f \"$f\" ] || continue; "
                  "p=0; grep -q PROBE_RESULT /root/probe_round_$n.log 2>/dev/null && p=1; "
                  "echo $n $p $(stat -c %s \"$f\"); done; true")
    if listing is None:
        return None
    for row in listing.splitlines():
        n, probed, size = row.split()
        if not n.isdigit():
            continue
        local = os.path.join(SAVES, f"{PREFIX}-{n}")
        lf = os.path.join(local, "adapter_model.safetensors")
        if not os.path.exists(lf) or os.path.getsize(lf) != int(size):
            os.makedirs(SAVES, exist_ok=True)
            subprocess.run(["scp", "-q", "-r", "-P", PORT, "-i", KEY,
                            f"{HOST}:/root/grpo_checkpoints/round-{n}/.", local], check=False)
            if not os.path.exists(lf):
                continue
            print(f"round-{n}: copied to {local}", flush=True)
        if probed == "1":
            remote_sha = (ssh(f"sha256sum /root/grpo_checkpoints/round-{n}/adapter_model.safetensors") or "").split(" ")[0]
            if remote_sha and remote_sha == sha(lf):
                ssh(f"rm -rf /root/grpo_checkpoints/round-{n}")
                print(f"round-{n}: probed and verified, removed from pod", flush=True)
            else:
                print(f"round-{n}: checksum mismatch, keeping pod copy", flush=True)
    return True


if __name__ == "__main__":
    idle = 0
    while True:
        ok = sync_once()
        alive = ssh("pgrep -f '^python3 /root/grpo_(combo|gentle)_train.py' | wc -l; ls /root/grpo_checkpoints | wc -l; true")
        if alive:
            running, left = (int(x) for x in alive.split()[:2])
            idle = idle + 1 if (running == 0 and left == 0) else 0
            if idle >= 3:
                print("trainer stopped and no checkpoints left on the pod; exiting", flush=True)
                break
        time.sleep(INTERVAL)
