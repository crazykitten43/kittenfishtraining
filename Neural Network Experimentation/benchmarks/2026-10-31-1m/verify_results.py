import csv, hashlib, json, sys
from pathlib import Path
import torch
root=Path('/home/kittenfishchess/kittenfish-experiments-20261031')
out=root/'benchmark-20261031'
sys.path.insert(0,str(root/'Neural Network Experimentation'))
from models import make_model
torch.set_num_threads(4)
summary={}
for name in ['resnet','transformer']:
    row=list(csv.DictReader((out/name/'metrics.csv').open()))
    assert len(row)==1 and int(row[0]['train_positions'])==1000000 and int(row[0]['val_positions'])==100000
    ckpts={}
    for kind in ['best','last']:
        p=out/name/(kind+'.pt')
        state=torch.load(p,map_location='cpu',weights_only=False)
        assert state['epoch']==0
        model=make_model(name).eval()
        model.load_state_dict(state['model'],strict=True)
        with torch.inference_mode():
            pred=model(torch.zeros(2,64,dtype=torch.uint8),torch.tensor([0,1]))
        assert torch.isfinite(pred).all()
        ckpts[kind]={'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'strict_reload_and_finite_inference':True}
    samples=list(csv.DictReader((out/(name+'-gpu.csv')).open()))
    memory_key=next(k for k in samples[0] if k.strip().startswith('memory.used'))
    driver_peak=max(float(r[memory_key].split()[0]) for r in samples)
    summary[name]={'metrics':row[0],'checkpoints':ckpts,'sampled_device_peak_mib':driver_peak}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))

import importlib.util
spec=importlib.util.spec_from_file_location('nnue_train',root/'v1/scripts/train_l4.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
state=torch.load(out/'nnue/model.last.pt',map_location='cpu',weights_only=False)
assert state['epoch']==1 and state['next_batch']==0
net=module.Net().eval()
net.load_state_dict(state['model'],strict=True)
import numpy as np
entry=json.loads((out/'manifest.json').read_text())['shards'][0]
rows=np.memmap(entry['path'],dtype=module.DTYPE,mode='r')[:2]
w,b,stm,_=module.features(rows)
with torch.inference_mode(): assert torch.isfinite(net(w,b,stm)).all()
ckpts={}
for p in sorted((out/'nnue').glob('*')):
    if p.suffix in ['.pt','.kff']:
        ckpts[p.name]={'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
summary['nnue']={'metrics':json.loads((out/'nnue/model.metrics.json').read_text()),'checkpoints':ckpts,'strict_reload_and_finite_inference':True}
samples=list(csv.DictReader((out/'nnue-gpu.csv').open()))
key=next(k for k in samples[0] if k.strip().startswith('memory.used'))
summary['nnue']['sampled_device_peak_mib']=max(float(r[key].split()[0]) for r in samples)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary['nnue'],indent=2))
