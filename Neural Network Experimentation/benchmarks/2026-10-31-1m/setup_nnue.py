import json, os
from pathlib import Path
root=Path('/home/kittenfishchess/kittenfish-experiments-20261031')
out=root/'benchmark-20261031'
m=json.loads((out/'manifest.json').read_text())
subset=out/'nnue-subset'
subset.mkdir(exist_ok=True)
for e in m['shards']:
    src=Path(e['path'])
    target=subset/src.name
    if not target.exists(): target.symlink_to(src)
    e['path']=src.name
(subset/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
p=root/'v1/scripts/train_l4.py'
s=p.read_text()
s=s.replace("        error = (net(w, b, stm) - target) * 400", "        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):\n            prediction = net(w, b, stm)\n        error = (prediction.float() - target) * 400")
s=s.replace("    net = Net().to(device)", "    net = Net().to(device)\n    parameter_count = sum(p.numel() for p in net.parameters())\n    print(f'NNUE: {parameter_count} stored parameters, {parameter_count-H} effective excluding padding; CUDA BF16 autocast', flush=True)")
s=s.replace("        started, seen = time.monotonic(), 0", "        if device.type == 'cuda':\n            torch.cuda.synchronize()\n            torch.cuda.reset_peak_memory_stats()\n        started, seen = time.monotonic(), 0")
s=s.replace("            loss = (net(w, b, stm) - target).square().mean()", "            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):\n                prediction = net(w, b, stm)\n                loss = (prediction.float() - target).square().mean()")
s=s.replace("        mse, mae = validate(net, loader(a, m, 'val', e), device)", "        if device.type == 'cuda': torch.cuda.synchronize()\n        train_seconds = time.monotonic() - started\n        mse, mae = validate(net, loader(a, m, 'val', e), device)\n        if device.type == 'cuda': torch.cuda.synchronize()\n        metrics = dict(epoch=e+1, train_positions=seen, val_positions=m['counts']['val'], train_seconds=train_seconds, train_positions_per_second=seen/train_seconds, seconds=time.monotonic()-started, val_mae_cp=mae, val_mse_cp2=mse, parameter_count=parameter_count, effective_parameter_count=parameter_count-H, peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else 0, peak_reserved_bytes=torch.cuda.max_memory_reserved() if device.type=='cuda' else 0, amp='bf16', batch_size=a.batch_size, lr=a.lr, seed=a.seed, dataset=identity)\n        Path(str(prefix)+'.metrics.json').write_text(json.dumps(metrics,indent=2)+'\\n')\n        print(json.dumps(metrics),flush=True)")
p.write_text(s)
