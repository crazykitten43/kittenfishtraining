import hashlib, json, random
from pathlib import Path
root = Path('/home/kittenfishchess/kittenfish-experiments-20261031')
out = root / 'benchmark-20261031'
out.mkdir(exist_ok=True)
source = Path('/home/kittenfishchess/kittenfishtraining-100m/data/kf100m/manifest.json')
meta = json.loads(source.read_text())
rng = random.Random(42)
selected = []
for split, n in [('train',10), ('val',1)]:
    entries = [e for e in meta['shards'] if e['split']==split and e['count']==100000]
    for e in rng.sample(entries,n):
        e = e.copy()
        p = source.parent / e['path']
        assert p.stat().st_size == e['count']*67
        assert hashlib.sha256(p.read_bytes()).hexdigest()==e['sha256']
        e['path']=str(p)
        selected.append(e)
meta['shards']=selected
meta['counts']={'train':1000000,'val':100000}
meta['subset_selection']={'seed':42,'method':'uniform sample of complete 100000-record shards without replacement','source_manifest':str(source)}
(out/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
p=root/'Neural Network Experimentation/train.py'
s=p.read_text()
s=s.replace("config['device_resolved'] = str(device)", "config['device_resolved'] = str(device)\n    config['parameter_count'] = sum(p.numel() for p in model.parameters())")
s=s.replace("        started = time.perf_counter()", "        if device.type == 'cuda':\n            torch.cuda.synchronize()\n            torch.cuda.reset_peak_memory_stats()\n        started = time.perf_counter()")
s=s.replace("        val_mse, val_mae, val_n = run('val', 0, a.max_val_batches)", "        if device.type == 'cuda':\n            torch.cuda.synchronize()\n        train_seconds = time.perf_counter() - started\n        train_peak_bytes = torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0\n        val_mse, val_mae, val_n = run('val', 0, a.max_val_batches)\n        if device.type == 'cuda':\n            torch.cuda.synchronize()")
s=s.replace("val_positions=val_n, seconds=elapsed)", "val_positions=val_n, seconds=elapsed, train_seconds=train_seconds,\n                   train_positions_per_second=train_n / train_seconds,\n                   parameter_count=config['parameter_count'],\n                   train_peak_allocated_bytes=train_peak_bytes,\n                   peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0,\n                   peak_reserved_bytes=torch.cuda.max_memory_reserved() if device.type == 'cuda' else 0)")
p.write_text(s)
print(json.dumps(meta['counts']))
