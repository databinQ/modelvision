# modelvision

Collect Neural Network components, includes:

- complete NN models
- special layers
- optimizers
- callbacks
- functions

Beside, some competition and project solutions are also included.

There are several purpose of this project:

- recording
- provide practical tools, which help to get a quick start doing something
- implementation of some papers

---

So far, following components have been implemented:

- Callback
    - [Snapshot Ensembles](https://arxiv.org/abs/1704.00109) at `callbacks.snapshot`
- Model
    - [DenseNet](https://arxiv.org/abs/1608.06993) at `models.densenet`
    - [Densely Interactive Inference Network(DIIN)](https://openreview.net/forum?id=r1dHXnH6-&noteId=r1dHXnH6-) at `models.diin`
    - [Transformer](https://arxiv.org/abs/1706.03762) at `models.tansformer`
- Layer
    - DecayingDropout at `layers.dropout`
    - [Transformer](https://arxiv.org/abs/1706.03762) at `layers.tansformer`
