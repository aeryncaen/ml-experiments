"""
Ripple vs baselines on MQAR — matching Zoology Figure 2 setup.

Runs Ripple alongside Attention and Mamba for direct comparison.
Sweeps: seq_len x d_model x lr (same grid as the paper).

Usage:
    python -m zoology.launch zoology/experiments/ripple_mqar.py
    python -m zoology.launch zoology/experiments/ripple_mqar.py -p  # parallel GPUs
"""
import uuid
import numpy as np
from zoology.config import TrainConfig, ModelConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig


sweep_id = uuid.uuid4().hex[:6]
sweep_name = f"ripple_mqar_{sweep_id}"

VOCAB_SIZE = 8_192

configs = []

for input_seq_len, num_kv_pairs in [
    (64, 4),
    (256, 16),
    (1024, 64),
    (4096, 256),
]:
    if input_seq_len <= 128:
        batch_size = 512
    elif input_seq_len <= 512:
        batch_size = 256
    elif input_seq_len <= 2048:
        batch_size = 128
    elif input_seq_len <= 4096:
        batch_size = 64
    else:
        batch_size = 32

    factory_kwargs = {
        "num_kv_pairs": num_kv_pairs,
        "train_power_a": 0.01,
        "test_power_a": 0.01,
        "random_non_queries": False,
    }

    data = DataConfig(
        train_configs=[MQARConfig(num_examples=100_000, vocab_size=VOCAB_SIZE, input_seq_len=input_seq_len, **factory_kwargs)],
        test_configs=[MQARConfig(num_examples=3_000, vocab_size=VOCAB_SIZE, input_seq_len=input_seq_len, **factory_kwargs)],
        batch_size=batch_size,
    )

    for d_model in [32, 64, 128, 256, 512]:
        for lr in np.logspace(-4, -2, 4)[1:]:

            # Determine num_heads: keep head_dim ~16-32 range
            if d_model <= 32:
                num_heads = 2
            elif d_model <= 64:
                num_heads = 4
            elif d_model <= 128:
                num_heads = 8
            elif d_model <= 256:
                num_heads = 8
            else:
                num_heads = 16

            MIXERS = {
                # Paper baselines
                "attention": dict(
                    name="zoology.mixers.attention.MHA",
                    kwargs={"dropout": 0.1, "num_heads": 1},
                ),
                "mamba": dict(
                    name="zoology.mixers.mamba.Mamba",
                    kwargs={},
                ),
                # Ripple variants
                "ripple-tcl": dict(
                    name="zoology.mixers.ripple.RippleMixer",
                    kwargs={
                        "num_heads": num_heads,
                        "order": "tele,conv,lowrank",
                        "max_seq_len": input_seq_len,
                    },
                ),
                "ripple-tclj": dict(
                    name="zoology.mixers.ripple.RippleMixer",
                    kwargs={
                        "num_heads": num_heads,
                        "order": "tele,conv,lowrank,jacobi",
                        "max_seq_len": input_seq_len,
                    },
                ),
                "ripple-aj": dict(
                    name="zoology.mixers.ripple.RippleMixer",
                    kwargs={
                        "num_heads": num_heads,
                        "order": "attn,jacobi",
                        "max_seq_len": input_seq_len,
                    },
                ),
                "ripple-ca": dict(
                    name="zoology.mixers.ripple.RippleMixer",
                    kwargs={
                        "num_heads": num_heads,
                        "order": "conv,attn",
                        "max_seq_len": input_seq_len,
                    },
                ),
                "ripple-cl": dict(
                    name="zoology.mixers.ripple.RippleMixer",
                    kwargs={
                        "num_heads": num_heads,
                        "order": "conv,lowrank",
                        "max_seq_len": input_seq_len,
                    },
                ),
            }

            for sequence_mixer in [
                "attention",
                "ripple-tcl",
                "ripple-ca",
                "ripple-cl",
            ]:
                if "mamba" in sequence_mixer:
                    block_type = "MambaBlock"
                else:
                    block_type = "TransformerBlock"

                model = ModelConfig(
                    d_model=d_model,
                    n_layers=2,
                    block_type=block_type,
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
