"""
ripple-msca vs attention across multiple tasks.

Tasks: forgetting_mqar, compositional_mqar, cumulative_parity

Usage:
    python -m zoology.launch zoology/experiments/ripple_tasks.py
    python -m zoology.launch zoology/experiments/ripple_tasks.py -p
"""
import uuid
import numpy as np
from zoology.config import TrainConfig, ModelConfig, DataConfig, LoggerConfig
from zoology.data.forgetting_mqar import ForgettingMQARConfig
from zoology.data.compositional_mqar import CompositionalMQARConfig
from zoology.data.circuits import CumulativeParityConfig

sweep_id = uuid.uuid4().hex[:6]
sweep_name = f"ripple_tasks_{sweep_id}"

VOCAB_SIZE = 8_192

configs = []

TASKS = {
    "forgetting": lambda sl, kv, bs: DataConfig(
        train_configs=[ForgettingMQARConfig(
            num_examples=100_000, vocab_size=VOCAB_SIZE, input_seq_len=sl,
            num_kv_pairs=kv, num_updates=max(1, kv // 2),
            power_a=0.01, random_non_queries=False,
        )],
        test_configs=[ForgettingMQARConfig(
            num_examples=3_000, vocab_size=VOCAB_SIZE, input_seq_len=sl,
            num_kv_pairs=kv, num_updates=max(1, kv // 2),
            power_a=0.01, random_non_queries=False,
        )],
        batch_size=bs,
    ),
    "compositional": lambda sl, kv, bs: DataConfig(
        train_configs=[CompositionalMQARConfig(
            num_examples=100_000, vocab_size=VOCAB_SIZE, input_seq_len=sl,
            num_kv_pairs=kv, power_a=0.01, random_non_queries=False,
        )],
        test_configs=[CompositionalMQARConfig(
            num_examples=3_000, vocab_size=VOCAB_SIZE, input_seq_len=sl,
            num_kv_pairs=kv, power_a=0.01, random_non_queries=False,
        )],
        batch_size=bs,
    ),
    "cum_parity": lambda sl, kv, bs: DataConfig(
        train_configs=[CumulativeParityConfig(
            num_examples=100_000, vocab_size=3, input_seq_len=sl,
        )],
        test_configs=[CumulativeParityConfig(
            num_examples=3_000, vocab_size=3, input_seq_len=sl,
        )],
        batch_size=bs,
    ),
}

SEQ_KV = [
    (64, 4),
    (256, 16),
    (512, 32),
    (1024, 64),
]
CUM_PARITY_SEQS = [16, 32, 64, 128]

for input_seq_len, num_kv_pairs in SEQ_KV:
    if input_seq_len <= 128:
        batch_size = 512
    elif input_seq_len <= 512:
        batch_size = 256
    elif input_seq_len <= 2048:
        batch_size = 128
    else:
        batch_size = 64

    # compositional needs perfect square kv pairs
    comp_kv = {4: 4, 16: 16, 32: 25, 64: 64}[num_kv_pairs]

    for d_model in [16, 32, 64, 128, 256]:
        for lr in np.logspace(-4, -2, 4)[2:]:

            if d_model <= 32:
                num_heads = 2
            elif d_model <= 64:
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
                "ripple-msca": dict(
                    name="zoology.mixers.ripple.RippleMixer",
                    kwargs={
                        "num_heads": num_heads,
                        "order": "msconv,attn",
                        "max_seq_len": input_seq_len,
                    },
                ),
                "ripple-conv3a": dict(
                    name="zoology.mixers.ripple.RippleMixer",
                    kwargs={
                        "num_heads": num_heads,
                        "order": "conv,attn",
                        "max_seq_len": input_seq_len,
                        "plain_conv_size": 3,
                    },
                ),
            }

            for task_name in ["forgetting", "compositional"]:
                data_fn = TASKS[task_name]
                kv = comp_kv if task_name == "compositional" else num_kv_pairs
                data = data_fn(input_seq_len, kv, batch_size)

                for mixer_name in ["attention", "ripple-msca", "ripple-conv3a"]:
                    model = ModelConfig(
                        d_model=d_model,
                        n_layers=2,
                        block_type="TransformerBlock",
                        max_position_embeddings=input_seq_len,
                        vocab_size=VOCAB_SIZE,
                        sequence_mixer=MIXERS[mixer_name],
                        state_mixer=dict(name="torch.nn.Identity", kwargs={}),
                    )
                    config = TrainConfig(
                        model=model,
                        data=data,
                        learning_rate=lr,
                        max_epochs=64,
                        run_id=f"{task_name}-{mixer_name}-seqlen{input_seq_len}-dmodel{d_model}-lr{lr:.6f}-kv{kv}",
                        logger=LoggerConfig(
                            project_name="ripple-tasks",
                        ),
                    )
                    configs.append(config)

# Cumulative parity: separate sweep with short seqlens + MLP
for input_seq_len in CUM_PARITY_SEQS:
    batch_size = 512

    for d_model in [16, 32, 64, 128, 256]:
        for lr in np.logspace(-4, -2, 4)[2:]:

            if d_model <= 32:
                num_heads = 2
            elif d_model <= 64:
                num_heads = 4
            elif d_model <= 128:
                num_heads = 8
            else:
                num_heads = 8

            MIXERS_CP = {
                "attention": dict(
                    name="zoology.mixers.attention.MHA",
                    kwargs={"dropout": 0.1, "num_heads": 1},
                ),
                "ripple-msca": dict(
                    name="zoology.mixers.ripple.RippleMixer",
                    kwargs={
                        "num_heads": num_heads,
                        "order": "msconv,attn",
                        "max_seq_len": input_seq_len,
                    },
                ),
                "ripple-conv3a": dict(
                    name="zoology.mixers.ripple.RippleMixer",
                    kwargs={
                        "num_heads": num_heads,
                        "order": "conv,attn",
                        "max_seq_len": input_seq_len,
                        "plain_conv_size": 3,
                    },
                ),
            }

            data = TASKS["cum_parity"](input_seq_len, 0, batch_size)

            for mixer_name in ["attention", "ripple-msca", "ripple-conv3a"]:
                if mixer_name.startswith("ripple"):
                    state_mixer = dict(name="zoology.mixers.mlp.LearnedGLU", kwargs={"hidden_mult": 4})
                else:
                    state_mixer = dict(name="zoology.mixers.mlp.GLU", kwargs={"hidden_mult": 4})
                model = ModelConfig(
                    d_model=d_model,
                    n_layers=2,
                    block_type="TransformerBlock",
                    max_position_embeddings=input_seq_len,
                    vocab_size=3,
                    sequence_mixer=MIXERS_CP[mixer_name],
                    state_mixer=state_mixer,
                )
                config = TrainConfig(
                    model=model,
                    data=data,
                    learning_rate=lr,
                    loss_scale=5.0,
                    max_epochs=64,
                    run_id=f"cum_parity-{mixer_name}-seqlen{input_seq_len}-dmodel{d_model}-lr{lr:.6f}",
                    logger=LoggerConfig(
                        project_name="ripple-tasks",
                    ),
                )
                configs.append(config)
