import math
import torch
import torch.distributed as dist
from itertools import chain
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.optim.optimizer import Optimizer, ParamsT
from typing import Callable, Generator, List, Optional, Tuple, Union

from dion.newton_schulz_triton import newton_schulz_triton, zeropower_via_newtonschulz5
from dion.opt_utils import (
    AsyncRuntime,
    AsyncTask,
    create_param_batches,
    pad_batch,
    to_local,
)
from dion.scalar_opts import adamw_update_foreach_async, lion_update_foreach_async

# Reuse Muon's helper functions
from dion.muon import (
    muon_update_newton_schulz,
)


# PolarExpress degree-5 coefficients from Amsel et al. (arXiv:2505.16932v3)
_PE_COEFFS_RAW = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.37500016454742485),
    (1.875, -1.25, 0.375),
]

_PE_COEFFS = [
    (a / 1.01, b / (1.01**3), c / (1.01**5))
    for (a, b, c) in _PE_COEFFS_RAW[:-1]
] + [_PE_COEFFS_RAW[-1]]


@torch.compile(dynamic=False, fullgraph=True, disable=not torch.cuda.is_available())
def zeropower_via_polarexpress5(G: Tensor, epsilon: float = 1e-7):
    """Turbo-Muon style AOL preconditioned PolarExpress.

    This applies AOL-like diagonal preconditioning from Turbo-Muon before
    the polynomial iterations, then runs 4 PolarExpress steps.
    """
    X = G.to(dtype=torch.bfloat16)
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT

    # Turbo-Muon AOL preconditioning (row-wise in the working orientation).
    # A0 is cached and reused to avoid recomputing X @ X^T on the first step.
    A = X @ X.mT
    s = torch.rsqrt(A.abs().sum(dim=-1, keepdim=True) + epsilon)
    X = X * s
    A = (A * s) * s.mT

    # Keep 5 PolarExpress iterations for this ablation.
    hs = _PE_COEFFS[:5]
    for i, (a, b, c) in enumerate(hs):
        if i > 0:
            A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.mT
    return X


class SpicyDion(Optimizer):
    """
    Distributed SpicyDion optimizer for PyTorch FSDP2. Also compatible with DDP.

    Args:
        params: Parameters for the optimizer.
        distributed_mesh: DeviceMesh or ProcessGroup for distributed training.
            Use DeviceMesh for FSDP2 and ProcessGroup for DistributedDataParallel.
        lr: Base learning rate. For Muon, this will be scaled based on the matrix dimensions.
            For element-wise update rules, this is the actual learning rate and no additional scaling is done.
        fraction: Fraction of submatrix to orthogonalize per update (0 < fraction <= 1).
        ef_decay: Error-feedback decay factor applied to selected submatrix.
        betas: Tuple of (beta1, beta2) for AdamW and Lion algorithms.
        weight_decay: Weight decay factor.
        epsilon: Small value to avoid division by zero.
        adjust_lr: How to adjust the learning rate for Muon updates ("spectral_norm" or "rms_norm" or None).
            "spectral_norm": Adjust based on spectral norm, for learning rate transfer across model scale.
            "rms_norm": Adjust based on RMS norm, for learning rate compatibility with Adam/AdamW.
            None: Do not adjust the learning rate.
        flatten: Whether to flatten 3D+ tensors to 2D for Muon updates.
            True: Tensors with 3+ dimensions are flattened to 2D. Use this for convolutional layers.
            False: Tensors are not flattened. 3D+ tensors are treated as batches of 2D matrices.
        use_triton: Whether to use Triton kernel for Newton-Schulz. Ignored if custom function is provided.
        newton_schulz_func: Use a custom Newton-Schulz function for orthogonalization.
            Signature is `func(input: Tensor, epsilon: float) -> Tensor`.
        verbose: Whether to print debug information during updates. If True, it prints whether rows or columns are selected for the submatrix selection process.

    SpicyDion optimizer (Dion2 baseline) by Ahn et al.: TBD
    """

    def __init__(
        self,
        params: ParamsT,
        distributed_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        lr: float = 0.01,
        fraction: float = 0.25,
        ef_decay: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        epsilon: float = 1e-8,
        adjust_lr: Optional[str] = "spectral_norm",
        flatten: bool = False,
        use_triton: bool = False,
        newton_schulz_func: Optional[Callable] = None,
        gnorm_beta: float = 0.55,
        total_steps: int = 1,
        verbose: bool = False,
    ):
        # Validate hyperparameters
        if total_steps < 1:
            raise ValueError(f"total_steps must be >= 1, got {total_steps}")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 < fraction <= 1.0):
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        if ef_decay < 0.0:
            raise ValueError(f"Invalid ef_decay: {ef_decay}")
        if len(betas) != 2 or betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid betas: {betas}")
        if adjust_lr not in ("spectral_norm", "rms_norm", None):
            raise ValueError(
                f"Invalid adjust_lr value: {adjust_lr}. Must be 'spectral_norm', 'rms_norm', or None."
            )
        if not (0.0 <= gnorm_beta <= 1.0):
            raise ValueError(f"gnorm_beta must be in [0, 1], got {gnorm_beta}")

        defaults = dict(
            lr=lr,
            ef_decay=ef_decay,
            fraction=fraction,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            epsilon=epsilon,
            flatten=flatten,
            adjust_lr=adjust_lr,
            algorithm="spicydion",
            step=0,
        )
        super().__init__(params, defaults)
        self.gnorm_beta = gnorm_beta
        self.total_steps = total_steps

        # Distributed configuration
        if isinstance(distributed_mesh, DeviceMesh):
            if distributed_mesh.ndim != 1:
                raise ValueError(
                    f"Only 1D DeviceMesh supported, but got {distributed_mesh.ndim}D. For HSDP, provide the 1D sharded sub-mesh."
                )
            self._device_rank = distributed_mesh.get_local_rank()
            self._world_size = distributed_mesh.size()
            self._process_group = distributed_mesh.get_group()
        elif isinstance(distributed_mesh, ProcessGroup):
            self._device_rank = dist.get_rank(distributed_mesh)
            self._world_size = dist.get_world_size(distributed_mesh)
            self._process_group = distributed_mesh
        elif distributed_mesh is None:
            self._device_rank = 0
            self._world_size = 1
            self._process_group = None
        else:
            raise TypeError(
                f"Invalid distributed_mesh type: {type(distributed_mesh)}. Expected DeviceMesh or ProcessGroup."
            )
        self._distributed_mesh = distributed_mesh

        # Orthogonalization configuration
        if newton_schulz_func is not None:
            if not callable(newton_schulz_func):
                raise TypeError(
                    f"newton_schulz_func must be a callable function, got {type(newton_schulz_func)}"
                )
            self._newton_schulz_func = newton_schulz_func
        elif use_triton:
            self._newton_schulz_func = newton_schulz_triton
        else:
            self._newton_schulz_func = zeropower_via_polarexpress5
        self.verbose = verbose

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        spicydion_groups = []
        lion_groups = []
        adamw_groups = []

        for group in self.param_groups:
            # Increment step
            group["step"] += 1

            # Split parameter groups by algorithm
            algo = group["algorithm"]
            if algo == "spicydion":
                spicydion_groups.append(group)
            elif algo == "lion":
                lion_groups.append(group)
            elif algo == "adamw":
                adamw_groups.append(group)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

        # Create async tasks for each algorithm
        spicydion_tasks = self._create_spicydion_tasks(spicydion_groups, verbose=self.verbose)
        lion_tasks = self._create_lion_tasks(lion_groups)
        adamw_tasks = self._create_adamw_tasks(adamw_groups)

        all_tasks = chain(spicydion_tasks, lion_tasks, adamw_tasks)
        runtime = AsyncRuntime(all_tasks, max_concurrent_tasks=3)
        runtime.run()

        # Expose the effective (pressure-scheduled) LR for external logging.
        # The training loop reads param_groups[0]["scheduled_lr"].
        for group in spicydion_groups:
            for p in group["params"]:
                if p in self.state and "gnorm_last_lr" in self.state[p]:
                    group["scheduled_lr"] = self.state[p]["gnorm_last_lr"].item()
                    break  # first param's median LR is representative

        return loss

    def _get_or_initialize_state(self, param: Tensor, algo: str) -> dict:
        """
        Get optimizer state for the given parameter tensor,
        or lazy-initialize it if it doesn't exist.
        """
        state = self.state[param]
        if not state:
            state["momentum"] = torch.zeros_like(param)
            if algo == "adamw":
                state["variance"] = torch.zeros_like(param)
        return state

    def _create_spicydion_tasks(
        self,
        param_groups: List[dict],
        verbose: bool = False,
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to create batches of SpicyDion matrices and generate
        AsyncTask objects so we can process multiple batches concurrently.
        """
        for group in param_groups:
            assert group["algorithm"] == "spicydion"
            assert all(
                p.ndim >= 2 for p in group["params"]
            ), "SpicyDion only supports matrix parameters."

            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            if "initial_lr" not in group:
                group["initial_lr"] = group["lr"]

            # Most hyperparameters as tensors for torch.compile
            # Here "fraction" only determines the dimension of the submatrix
            # to be orthonormalized. Hence, it doesn't need to be a tensor
            spicydion_args = dict(
                lr=torch.tensor(group["lr"]),
                initial_lr=torch.tensor(group["initial_lr"]),
                step=torch.tensor(group["step"]),
                total_steps=torch.tensor(self.total_steps),
                gnorm_beta=torch.tensor(self.gnorm_beta),
                ef_decay=torch.tensor(group["ef_decay"]),
                fraction=group["fraction"],
                weight_decay=torch.tensor(group["weight_decay"]),
                epsilon=torch.tensor(group["epsilon"]),
                flatten=group["flatten"],
                adjust_lr=group["adjust_lr"],
                device_rank=self._device_rank,
                world_size=self._world_size,
                process_group=self._process_group,
                newton_schulz_func=self._newton_schulz_func,
            )

            # Create batches of parameters of size self._world_size
            for params in create_param_batches(
                group_params, batch_size=self._world_size
            ):
                gradients = [p.grad for p in params]
                states = [self._get_or_initialize_state(p, "spicydion") for p in params]
                momentums = [s["momentum"] for s in states]

                # Get sharding state for DTensor
                is_batch_sharded = False
                is_matrix_sharded = False
                sharded_mesh_dim = None
                sharded_tensor_dim = None

                if isinstance(params[0], DTensor):
                    if not isinstance(self._distributed_mesh, DeviceMesh):
                        raise RuntimeError(
                            "Must create optimizer with DeviceMesh if using DTensor parameters."
                        )

                    # Find the sharded placement and get its mesh and tensor dimensions
                    # Skip any Shard() placements on size-1 mesh dimension = Replicate()
                    shard_placements = [
                        (i, p)
                        for i, p in enumerate(params[0].placements)
                        if p.is_shard() and params[0].device_mesh.size(i) > 1
                    ]

                    # If we don't flatten 3D matrices, we can ignore shard placements along batch dimensions
                    # Only keep placements that shard one of the two matrix dimensions
                    if not group["flatten"]:
                        matrix_dims = {params[0].ndim - 1, params[0].ndim - 2}
                        is_batch_sharded = any(
                            p.dim not in matrix_dims for _, p in shard_placements
                        )
                        shard_placements = [
                            (i, p) for i, p in shard_placements if p.dim in matrix_dims
                        ]

                    # Check that we have no more than 1 sharded matrix dimension
                    # Note that non-flattened 3D tensors can have additional sharded batch dimensions
                    # Flattened 3D tensors are limited to one sharded dimension out of all dimensions
                    if len(shard_placements) == 1:
                        is_matrix_sharded = True
                        sharded_mesh_dim = shard_placements[0][0]
                        sharded_tensor_dim = shard_placements[0][1].dim
                    elif len(shard_placements) > 1:
                        raise NotImplementedError(
                            "SpicyDion does not support parameters with multiple sharded dimensions."
                        )

                    # Check that the sharded mesh dimension matches optimizer's device mesh
                    if (
                        sharded_mesh_dim is not None
                        and params[0].device_mesh.get_group(sharded_mesh_dim)
                                != self._process_group
                    ):
                        raise RuntimeError(
                            f"Got DTensor sharded over mesh dimension {sharded_mesh_dim} different from the optimizer's device mesh. "
                            f"DTensor has mesh: {params[0].device_mesh}, placements: {params[0].placements}, but optimizer was created with mesh: {self._distributed_mesh}."
                        )

                gnorm_ema = []
                gnorm_max = []
                gnorm_last_lr = []
                gnorm_last_units = []
                gnorm_last_signal = []
                gnorm_last_max = []
                gnorm_last_ratio = []
                for p, s in zip(params, states):
                    # Track gnorm per output neuron (row dimension), matching AutoNorMuon.
                    n_units = p.shape[-2] if p.ndim >= 2 else 1
                    cur_ema = s.get("gnorm_ema")
                    cur_max = s.get("gnorm_max")
                    if not isinstance(cur_ema, torch.Tensor) or cur_ema.shape != (n_units,):
                        cur_ema = torch.zeros((n_units,), device=p.device, dtype=torch.float32)
                        s["gnorm_ema"] = cur_ema
                    if not isinstance(cur_max, torch.Tensor) or cur_max.shape != (n_units,):
                        cur_max = torch.zeros((n_units,), device=p.device, dtype=torch.float32)
                        s["gnorm_max"] = cur_max
                    cur_last_lr = s.get("gnorm_last_lr")
                    if not isinstance(cur_last_lr, torch.Tensor) or cur_last_lr.numel() != 1:
                        cur_last_lr = torch.zeros((), device=p.device, dtype=torch.float32)
                        s["gnorm_last_lr"] = cur_last_lr
                    cur_last_units = s.get("gnorm_last_units")
                    if not isinstance(cur_last_units, torch.Tensor) or cur_last_units.numel() != 1:
                        cur_last_units = torch.zeros((), device=p.device, dtype=torch.float32)
                        s["gnorm_last_units"] = cur_last_units
                    cur_last_signal = s.get("gnorm_last_signal")
                    if not isinstance(cur_last_signal, torch.Tensor) or cur_last_signal.numel() != 1:
                        cur_last_signal = torch.zeros((), device=p.device, dtype=torch.float32)
                        s["gnorm_last_signal"] = cur_last_signal
                    cur_last_max = s.get("gnorm_last_max")
                    if not isinstance(cur_last_max, torch.Tensor) or cur_last_max.numel() != 1:
                        cur_last_max = torch.zeros((), device=p.device, dtype=torch.float32)
                        s["gnorm_last_max"] = cur_last_max
                    cur_last_ratio = s.get("gnorm_last_ratio")
                    if not isinstance(cur_last_ratio, torch.Tensor) or cur_last_ratio.numel() != 1:
                        cur_last_ratio = torch.zeros((), device=p.device, dtype=torch.float32)
                        s["gnorm_last_ratio"] = cur_last_ratio
                    gnorm_ema.append(cur_ema)
                    gnorm_max.append(cur_max)
                    gnorm_last_lr.append(cur_last_lr)
                    gnorm_last_units.append(cur_last_units)
                    gnorm_last_signal.append(cur_last_signal)
                    gnorm_last_max.append(cur_last_max)
                    gnorm_last_ratio.append(cur_last_ratio)

                # Special case for 3D tensors sharded along batch dimension
                # As long as matrix dimensions are not sharded, each device will have whole matrices
                # Each device already has different matrices of the batch, so we can't parallelize further
                if is_batch_sharded and not is_matrix_sharded:
                    for x, g, m, ge, gm, gl, gu, gs, gx, gr in zip(
                        params,
                        gradients,
                        momentums,
                        gnorm_ema,
                        gnorm_max,
                        gnorm_last_lr,
                        gnorm_last_units,
                        gnorm_last_signal,
                        gnorm_last_max,
                        gnorm_last_ratio,
                    ):
                        yield AsyncTask(
                            spicydion_update_batch_async(
                                X=[x],
                                G=[g],
                                M=[m],
                                GNORM_EMA=[ge],
                                GNORM_MAX=[gm],
                                GNORM_LAST_LR=[gl],
                                GNORM_LAST_UNITS=[gu],
                                GNORM_LAST_SIGNAL=[gs],
                                GNORM_LAST_MAX=[gx],
                                GNORM_LAST_RATIO=[gr],
                                shard_dim=None,  # No sharded matrix dim
                                **spicydion_args,
                                verbose=verbose,
                            )
                        )
                # Otherwise, we parallelize the Muon update across devices
                else:
                    yield AsyncTask(
                        spicydion_update_batch_async(
                            X=pad_batch(params, self._world_size),
                            G=pad_batch(gradients, self._world_size),
                            M=pad_batch(momentums, self._world_size),
                            GNORM_EMA=pad_batch(gnorm_ema, self._world_size),
                            GNORM_MAX=pad_batch(gnorm_max, self._world_size),
                            GNORM_LAST_LR=pad_batch(gnorm_last_lr, self._world_size),
                            GNORM_LAST_UNITS=pad_batch(gnorm_last_units, self._world_size),
                            GNORM_LAST_SIGNAL=pad_batch(gnorm_last_signal, self._world_size),
                            GNORM_LAST_MAX=pad_batch(gnorm_last_max, self._world_size),
                            GNORM_LAST_RATIO=pad_batch(gnorm_last_ratio, self._world_size),
                            shard_dim=sharded_tensor_dim,
                            **spicydion_args,
                            verbose=verbose,
                        )
                    )

    def _create_lion_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "lion",
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to generate AsyncTask objects for Lion updates.
        """
        for group in param_groups:
            assert group["algorithm"] == algo_name

            # Get parameters and optimizer states
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, algo_name) for p in params]
            momentums = [s["momentum"] for s in states]

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])

            yield AsyncTask(
                lion_update_foreach_async(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                )
            )

    def _create_adamw_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "adamw",
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to generate AsyncTask objects for AdamW updates.
        """
        for group in param_groups:
            assert group["algorithm"] == algo_name

            # Get parameters and optimizer states
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, algo_name) for p in params]
            momentums = [s["momentum"] for s in states]
            variances = [s["variance"] for s in states]

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])
            epsilon = torch.tensor(group["epsilon"])
            step = torch.tensor(group["step"])

            yield AsyncTask(
                adamw_update_foreach_async(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    V=to_local(variances),
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                    step=step,
                    epsilon=epsilon,
                )
            )


def spicydion_update_batch_async(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    GNORM_EMA: List[Tensor],  # Per-unit EMA gnorm state (modified in place)
    GNORM_MAX: List[Tensor],  # Per-unit max gnorm state (modified in place)
    GNORM_LAST_LR: List[Tensor],  # Last applied median LR (modified in place)
    GNORM_LAST_UNITS: List[Tensor],  # Last units median (EMA input)
    GNORM_LAST_SIGNAL: List[Tensor],  # Last signal median (EMA output)
    GNORM_LAST_MAX: List[Tensor],  # Last denominator median
    GNORM_LAST_RATIO: List[Tensor],  # Last ratio median
    lr: Tensor,  # Learning rate (scalar tensor)
    initial_lr: Tensor,  # Base (unscheduled) learning rate
    step: Tensor,  # Optimizer step
    total_steps: Tensor,  # Total training steps (for cosine ceiling)
    gnorm_beta: Tensor,  # EMA beta for gnorm
    ef_decay: Tensor,  # Error-feedback factor (scalar tensor)
    fraction: float,  # Fraction of submatrix to orthogonalize (0 < fraction <= 1)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    epsilon: Tensor,  # Epsilon (scalar tensor)
    flatten: bool,  # Whether to flatten 3D+ tensors to 2D
    adjust_lr: Optional[str],  # How to adjust learning rate
    device_rank: int,  # Rank of the current device
    world_size: int,  # Total number of devices to parallelize over
    shard_dim: Optional[int] = None,  # Shard dimension for DTensor (if applicable)
    process_group: Optional[ProcessGroup] = None,
    newton_schulz_func: Optional[Callable] = None,
    verbose: bool = False,
) -> Generator[None, None, None]:
    """
    Batched version of SpicyDion update. Batch size should be equal to number of GPUs.
    All tensors in a batch should have identical shape, sharding, and dtype.
    Identical hyperparameters are used for all tensors in the batch.
    """
    assert len(X) == len(G)
    assert len(X) == len(M)
    assert len(X) == len(GNORM_EMA)
    assert len(X) == len(GNORM_MAX)
    assert len(X) == len(GNORM_LAST_LR)
    assert len(X) == len(GNORM_LAST_UNITS)
    assert len(X) == len(GNORM_LAST_SIGNAL)
    assert len(X) == len(GNORM_LAST_MAX)
    assert len(X) == len(GNORM_LAST_RATIO)

    # Determine selection dimension based on sharding and tensor shape:
    # For sharded matrices, we align select_dim with shard_dim
    # For unsharded matrices (DDP or single-GPU), we select the shorter dimension
    ndim = X[0].ndim
    select_dim = None

    if shard_dim is not None:
        # Normalize shard_dim to negative indexing for unified treatment
        shard_dim = shard_dim if shard_dim < 0 else shard_dim - ndim
        if shard_dim == -2:
            select_dim = -2  # Row-sharded
        elif shard_dim == -1:
            select_dim = -1  # Column-sharded

    # Fall-back to shorter dimension when DDP, Single-GPU, or batch-sharded
    if select_dim is None:
        num_rows, num_cols = X[0].shape[-2:]
        select_dim = -2 if num_rows <= num_cols else -1

    # Print how the selection choice based on shard_dim and tensor shape
    if verbose:
        _print_selection_choice(X[0].shape, shard_dim, select_dim, ndim)

    # Update momentum and select top-α fraction along select_dim
    U_selected, indices_list = spicydion_pre_orthogonalize(
        G=to_local(G),
        M=to_local(M),
        fraction=fraction,
        ef_decay=ef_decay,
        select_dim=select_dim,
    )

    # Get one whole matrix for each device to orthogonalize
    if shard_dim is not None:
        # Use all-to-all to transform from a batch of shards to a single whole matrix
        # https://www.essential.ai/blog/infra
        assert len(X) == world_size, "Batch size must equal world size"
        assert (
            process_group is not None
        ), "process_group must be provided for sharded DTensors"
        assert isinstance(X[0], DTensor), "X should contain DTensors"
        assert (
            X[0].size(shard_dim) % world_size == 0
        ), f"Shard dimension {shard_dim} size {X[0].size(shard_dim)} is not divisible by world size {world_size}."

        # Allocate buffers to receive shards of one whole submatrix from other devices
        recv_shards = [torch.empty_like(u) for u in U_selected]
        work = dist.all_to_all(
            recv_shards, U_selected, group=process_group, async_op=True
        )
        yield
        work.wait()

        # Concatentate shards to form a whole matrix to orthogonalize
        # Only submatrix is orthogonalized!
        full_submatrix = torch.cat(recv_shards, dim=select_dim)
        full_submatrix = muon_update_newton_schulz(
            full_submatrix, newton_schulz_func, flatten=flatten, epsilon=epsilon
        )

        # Split result back into shards
        # Contiguous is needed for all-to-all to work correctly
        send_shards = [
            t.contiguous()
            for t in torch.tensor_split(full_submatrix, world_size, dim=select_dim)
        ]

        # Redistribute the orthogonalized tensor back to original layout
        U_ortho = [torch.empty_like(u) for u in U_selected]
        work = dist.all_to_all(U_ortho, send_shards, group=process_group, async_op=True)
        yield
        work.wait()

    # Matrices are not sharded, so we can distribute the batch across different devices
    # Get a single matrix of the batch corresponding to this device
    elif len(U_selected) > 1:
        assert len(U_selected) == world_size, "Batch size must equal world size"
        assert process_group is not None

        single_matrix = U_selected[device_rank]
        assert not isinstance(single_matrix, DTensor)

        single_ortho = muon_update_newton_schulz(
            single_matrix,
            newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
        )

        # Allocate empty tensors to receive updates from other devices
        U_ortho = [torch.empty_like(u) for u in U_selected]
        # All gather orthogonalized results from other devices into buffer
        work = dist.all_gather(
            U_ortho, single_ortho.contiguous(), group=process_group, async_op=True
        )
        yield
        work.wait()

    # Single tensor with no sharded dimension. This happens in 2 cases:
    # - Running on a single GPU
    # - 3D+ tensors sharded along a batch dimension (different whole matrices per device)
    else:
        assert len(U_selected) == 1
        U_ortho = [
            muon_update_newton_schulz(
                U_selected[0], newton_schulz_func, flatten=flatten, epsilon=epsilon
            )
        ]

    # Update model parameters with orthogonalized output
    # Weight update is applied to selected slices only
    spicydion_post_orthogonalize(
        X=to_local(X),
        G=to_local(G),
        U=U_ortho,
        indices=indices_list,
        GNORM_EMA=to_local(GNORM_EMA),
        GNORM_MAX=to_local(GNORM_MAX),
        GNORM_LAST_LR=to_local(GNORM_LAST_LR),
        GNORM_LAST_UNITS=to_local(GNORM_LAST_UNITS),
        GNORM_LAST_SIGNAL=to_local(GNORM_LAST_SIGNAL),
        GNORM_LAST_MAX=to_local(GNORM_LAST_MAX),
        GNORM_LAST_RATIO=to_local(GNORM_LAST_RATIO),
        base_lr=initial_lr,
        step=step,
        total_steps=total_steps,
        gnorm_beta=gnorm_beta,
        weight_decay=weight_decay,
    )


@torch.compile(fullgraph=True, disable=not torch.cuda.is_available())
def spicydion_pre_orthogonalize(
    G: List[Tensor],
    M: List[Tensor],
    fraction: Tensor,
    ef_decay: Tensor,
    select_dim: int,
) -> Tuple[List[Tensor], List[Tensor]]:
    """
    EMA momentum + Nesterov, matching NorMuon/AutoNorMuon semantics.
    ef_decay is used as the momentum beta (e.g. 0.95).

    M.lerp_(G, 1 - beta)          =>  M = beta*M + (1-beta)*G
    update = lerp(G, M, beta)      =>  Nesterov extrapolation

    Inputs and outputs should be lists of regular Tensor, not DTensor.
    This is a separate function for compatibility with torch.compile().
    """
    dtype = M[0].dtype
    beta = ef_decay  # reuse ef_decay as momentum beta

    num_select = M[0].size(select_dim)

    # EMA momentum update: M = beta*M + (1-beta)*G
    G = [g.to(dtype=dtype) for g in G]
    torch._foreach_lerp_(M, G, 1.0 - beta)

    # Nesterov extrapolation: update = lerp(G, M, beta)
    beta_val = beta.item() if isinstance(beta, Tensor) else float(beta)
    updates = [torch.lerp(g, m, beta_val) for g, m in zip(G, M)]

    U_stacked = torch.stack(updates, dim=0)

    # No sparsity: always use the full matrix.
    batch_size = U_stacked.size(0)
    indices = torch.arange(num_select, device=U_stacked.device, dtype=torch.long)
    indices = indices.unsqueeze(0).expand(batch_size, -1)

    indices_list = list(indices.unbind(dim=0))

    # Convert to bf16 and unstack for communication
    U_selected = list(U_stacked.to(dtype=torch.bfloat16).unbind(dim=0))

    return U_selected, indices_list


@torch.compile(fullgraph=True, disable=not torch.cuda.is_available())
def spicydion_post_orthogonalize(
    X: List[Tensor],
    G: List[Tensor],
    U: List[Tensor],
    indices: List[Tensor],
    GNORM_EMA: List[Tensor],
    GNORM_MAX: List[Tensor],
    GNORM_LAST_LR: List[Tensor],
    GNORM_LAST_UNITS: List[Tensor],
    GNORM_LAST_SIGNAL: List[Tensor],
    GNORM_LAST_MAX: List[Tensor],
    GNORM_LAST_RATIO: List[Tensor],
    base_lr: Tensor,
    step: Tensor,
    total_steps: Tensor,
    gnorm_beta: Tensor,
    weight_decay: Tensor,
):
    """
    Apply weight update after orthogonalization.
    Per-neuron LR via pressure-based force balance:
        pressure = k * (step / total_steps)
        lr_neuron = base_lr * ratio / (ratio + pressure)
    Each neuron's gnorm ratio resists the external pressure independently.
    """
    # Pressure increases linearly over training budget. k controls max pressure.
    k = 4.0
    pressure = k * ((step - 1) / total_steps)

    # Convert U to match parameter dtype
    dtype = X[0].dtype
    U = [u.to(dtype=dtype) for u in U]
    is_first = (step <= 1)

    for x, g, u, idx, g_ema, g_max, g_last_lr, g_last_units, g_last_signal, g_last_max, g_last_ratio in zip(
        X,
        G,
        U,
        indices,
        GNORM_EMA,
        GNORM_MAX,
        GNORM_LAST_LR,
        GNORM_LAST_UNITS,
        GNORM_LAST_SIGNAL,
        GNORM_LAST_MAX,
        GNORM_LAST_RATIO,
    ):
        # Aspect-ratio scaling, matching AutoNorMuon/NorMuon (autonormuon.py:62).
        aspect = max(1, u.shape[-2] / u.shape[-1]) ** 0.5
        u = u * aspect

        # Post-ortho update norms for gnorm tracking.
        rows = u.float().reshape(u.shape[0], -1)
        units = rows.norm(dim=-1)

        prev_ema = g_ema
        prev_max = g_max

        ema_candidate = gnorm_beta * prev_ema + (1 - gnorm_beta) * units
        signal_u = torch.where(is_first, units, ema_candidate)
        max_candidate = torch.maximum(prev_max, units)
        max_u = torch.where(is_first, units, max_candidate)

        g_ema.copy_(signal_u.detach())
        g_max.copy_(max_u.detach())

        # Per-neuron ratio: how strong is this neuron's gradient signal vs its peak?
        ratio_u = signal_u / max_u.clamp(min=1e-12)

        # Force balance: ratio resists external pressure.
        # lr = base_lr * ratio / (ratio + pressure), floored at 10% of base_lr
        lr_u = base_lr * ratio_u / (ratio_u + pressure).clamp(min=1e-12)
        lr_u = torch.clamp(lr_u, min=1e-3 * base_lr.item())

        lr_shape = (lr_u.shape[0],) + (1,) * (u.ndim - 1)
        u_scaled = -(u * lr_u.to(dtype=u.dtype).view(lr_shape))

        g_last_lr.copy_(lr_u.median().detach())
        g_last_units.copy_(units.median().detach())
        g_last_signal.copy_(signal_u.median().detach())
        g_last_max.copy_(max_u.median().detach())
        g_last_ratio.copy_(ratio_u.median().detach())

        x.add_(u_scaled)

        # Retract weights to unit sphere (unit row norms).
        x.copy_(x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8))


# A helper function to print selection chocie for each matrix
# It only prints once `verbose` is set True
_printed_configs: set = set()


def _print_selection_choice(
    shape: torch.Size,
    shard_dim: Optional[int],
    select_dim: int,
    ndim: int,
):
    config_key = (tuple(shape), shard_dim, select_dim)
    if config_key not in _printed_configs:
        _printed_configs.add(config_key)

        num_rows, num_cols = shape[-2:]
        select_info = "rows" if select_dim == -2 else "columns"
        norm_info = "row norms" if select_dim == -2 else "col norms"

        if shard_dim is None:
            mode = "DDP/Single-GPU"
            shorter = "rows" if num_rows <= num_cols else "cols"
            reason = f"shorter dim = {shorter} ({min(num_rows, num_cols)})"
        else:
            # Normalize shard_dim for display
            normalized = shard_dim if shard_dim < 0 else shard_dim - ndim
            if normalized == -2:
                mode = "FSDP"
                reason = f"row-sharded (shard_dim={shard_dim}→-2)"
            elif normalized == -1:
                mode = "FSDP"
                reason = f"col-sharded (shard_dim={shard_dim}→-1)"
            else:
                mode = "FSDP batch-sharded"
                shorter = "rows" if num_rows <= num_cols else "cols"
                reason = f"shard_dim={shard_dim} (batch), shorter = {shorter}"

        print(
            f"[SpicyDion] Shape {tuple(shape)}: {mode}, {reason} → "
            f"select top-α {select_info} by {norm_info}"
        )
