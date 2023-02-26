import torch
from torch import nn
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs import Compose, DoubleToFloat, ObservationNorm, StepCounter, TransformedEnv
from torchrl.envs.utils import check_env_specs

device = "cpu" if not torch.has_cuda else "cuda:0"
num_cells = 256
lr = 3e-4
max_grad_norm = 1.0

frame_skip = 1
frames_per_batch = 1000 // frame_skip
total_frames = 50000 // frame_skip

sub_batch_size = 64
num_epoches = 10
clip_epsilon = (
    0.2
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

base_env = GymEnv("InvertedDoublePendulum-v4", device=device, frame_skip=frame_skip)

env = TransformedEnv(
    base_env,
    Compose(
        ObservationNorm(in_keys=["observation"]),
        DoubleToFloat(
            in_kes=["observation"],
        ),
        StepCounter(),
    ),
)

env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

check_env_specs(env)

actor_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.Lazy
)