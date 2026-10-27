# KittenFish v0 training

CNN training and Stockfish position generation moved from kittenfishfullstack.

```sh
cd v0
python -m pip install -r requirements.txt
python train.py --help
python PositionGeneration.py --help
```

Default dataset: `data/positions_15k.csv.gz`. Supply `--data` for another file.
Training preserves the original side-to-move score convention and game-grouped split.
The deployed engine and checkpoint live in [kittenfishv0](https://github.com/crazykitten43/kittenfishv0).
