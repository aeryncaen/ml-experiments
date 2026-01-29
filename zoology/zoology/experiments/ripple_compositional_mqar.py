"""
Ripple vs baselines on Compositional MQAR.

Compound keys (K1, K2) -> V. Neither key alone determines value,
forcing true compositional binding. num_kv_pairs must be perfect squares.

Usage:
    python -m zoology.launch zoology/experiments/ripple_compositional_mqar.py
    python -m zoology.launch zoology/experiments/ripple_compositional_mqar.py -p
"""
import uuid
import numpy as np
from zoology.config import TrainConfig, ModelConfig, DataConfig, LoggerConfig
from zoology.data.compositional_mqar import CompositionalMQARConfig

sweep_id = uuid.uuid4().hex[:6]
sweep_name = f"ripple_comp_mqar_{sweep_id}"

VOCAB_SIZE = 8_192

configs = []

for input_seq_len, num_kv_pairs in [
    (64, 4),       # 2x2 grid
    (128, 9),      # 3x3 grid
    (256, 16),     # 4x4 grid
    (512, 36),     # 6x6 grid
    (1024, 64),    # 8x8 grid
]:
    if input_seq_len <= 128:
        batch_size = 512
    elif input_seq_len <= 512:
        batch_size = 256
    elif input_seq_len <= 2048:
        batch_size = 128
    else:
        batch_size = 64

    factory_kwargs = {
        "num_kv_pairs": num_kv_pairs,
        "power_a": 0.01,
        "random_non_queries": False,
    }

    data = DataConfig(
        train_configs=[CompositionalMQARConfig(num_examples=100_000, vocab_size=VOCAB_SIZE, input_seq_len=input_seq_len, **factory_kwargs)],
        test_configs=[CompositionalMQARConfig(num_examples=3_000, vocab_size=VOCAB_SIZE, input_seq_len=input_seq_len, **factory_kwargs)],
        batch_size=batch_size,
    )

    for d_model in [64, 128, 256]:
        for lr in np.logspace(-4, -2, 4)[1:]:

            if d_model <= 64:
                num_heads = 4
            elif d_model <= 128:
                num_heads = 8
            else:
                num_heads = 8

            MIXERS = {
                "attention": dict(
                    name="zoology.mixers.attention.MHA",
                    kwargs={"dropout": 0.1, "num_heads": 1},
                ),
                "ripple-ja-dir-bcn": dict(
                    name="zoology.mixers.ripple.RippleMixer",
                    kwargs={
                        "num_heads": num_heads,
                        "order": "jacobi,attn",
                        "max_seq_len": input_seq_len,
                        "diffuse_se": True,
                        "diff_inject": True,
                        "diff_readout": True,
                        "bc_norm": True,
                    },
                ),

            }

            for sequence_mixer in [
                "attention",
                "ripple-ja-dir-bcn",
            ]:
                model = ModelConfig(
                    d_model=d_model,
                    n_layers=2,
                    block_type="TransformerBlock",
                    max_position_embeddings=input_seq_len if sequence_mixer == "attention" else 0,
                    vocab_size=VOCAB_SIZE,
                    sequence_mixer=MIXERS[sequence_mixer],
                    state_mixer=dict(name="torch.nn.Identity", kwargs={}),
                )
                config = TrainConfig(
                    model=model,
                    data=data,
                    learning_rate=lr,
                    max_epochs=64,
                    run_id=f"{sequence_mixer}-seqlen{input_seq_len}-dmodel{d_model}-lr{lr:.6f}-kv{num_kv_pairs}",
                    logger=LoggerConfig(
                        project_name="ripple-zoology",
                    ),
                )
                configs.append(config)
