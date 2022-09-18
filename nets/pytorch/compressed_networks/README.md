# Compressed Networks

These are checkpoints for networks that were compressed using [torchprune](https://github.com/lucaslie/torchprune)

For retraining and training we used recipes (optimizers, schedulers etc.) obtained from this repo <https://github.com/clovaai/frostnet>. These were baked into `torchprune`.

The compressing procedure was cascade, which retrained and compressed the network for 10 iterations using these keep ratios:

```python
np.geomspace(0.12, 0.85, 10)

>>> array([0.12, 0.14915991, 0.18540564, 0.23045907, 0.28646044, 0.3560701, 0.44259486, 0.55014506, 0.68382988, 0.85])
```

### Loading the compressed network using a checkpoint

To load a provided checkpoint, install the `torchprune` package, then the process of loading a network is shown in `get_network.py`
