# KittenFish v0.5 training

L4 SE-ResNet training moved from the `l4-resnet-v1` and `l4-beam-search-v1`
branches of kittenfishfullstack.

```sh
cd v0.5
python -m pip install -r requirements.txt
python train_l4.py --help
python train_l4.py --data /path/to/positions_100k.csv.gz --output checkpoints/kittenfish_l4_resnet.pt
python evaluate.py --data /path/to/positions_100k.csv.gz --checkpoint checkpoints/kittenfish_l4_resnet.pt
python test_model.py
```

`model.py` and `encoding.py` are the original L4 SE-ResNet and 18-plane encoder.
`resnettrain.py` preserves the earlier plain ResNet trainer and imports this
same compatible encoder. `PositionGeneration.py` is shared with the CNN generation.
Use explicit dataset paths; original defaults are retained.
The deployed checkpoint, searches and UCI engine live in
[kittenfishv0.5](https://github.com/crazykitten43/kittenfishv0.5).
