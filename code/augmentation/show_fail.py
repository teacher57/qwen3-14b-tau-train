import json, sys
sys.path.insert(0, '.')
import classify_failures as C
D = 'data/'
T = {}
for l in open(D + 'retail_aug_4trials_transcripts.jsonl'):
    c = json.loads(l); T[(c['task_index'], c['trial'])] = c
def short(x, n): x = ' '.join(str(x).split()); return x if len(x) <= n else x[:n] + '…'
def show(task, trial, ulim=170, alim=130):
    c = T[(task, trial)]; t = C.TASKS[task]; succ, errs, opened, tr, au = C.parse(c)
    tgt = [(a['name'], a['kwargs']) for a in t['actions'] if a['name'] in C.WRITE]
    print('=' * 100); print(f'TASK {task} trial {trial} reward {c["reward"]} | cause {C.classify(c, t)[0]}')
    print('WANT:', short(t['instruction'], 330))
    print('TARGET:', [(n, {k: v for k, v in a.items() if k in ('order_id', 'item_ids', 'new_item_ids', 'payment_method_id', 'reason')}) for n, a in tgt])
    print('DONE  :', [(n, {k: v for k, v in a.items() if k in ('order_id', 'item_ids', 'new_item_ids', 'payment_method_id', 'reason')}) for n, a in succ], '| errors:', [short(e, 60) for e in errs][:2], '| transferred', tr)
    for m in c['messages']:
        if m['role'] == 'user': print('U:', short(m['content'], ulim))
        elif m['role'] == 'assistant' and (m.get('content') or '').strip(): print('A:', short(m['content'], alim))
if __name__ == '__main__':
    auto = json.load(open(D + 'failure_causes_auto.json'))['failures']['augmented']
    cause = int(sys.argv[1]); k = int(sys.argv[2]); skip = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    sel = [r for r in auto if r['cause'] == cause][skip:skip + k]
    for r in sel: show(r['task'], r['trial'])
