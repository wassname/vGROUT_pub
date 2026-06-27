set shell := ["bash", "-cu"]

TRAIN := "uv run python -m vgrout.train"
TEACHER_POOL := "data/pools/teacher_pool"

default:
    @just --list

# Correctness gate: tiny-random Qwen3 on CPU, BEARTYPE on, routeA by default.
smoke *ARGS:
    uv run python scripts/verify_rewards.py
    uv run python scripts/verify_eval_gap.py
    uv run python scripts/verify_partition.py
    uv run python scripts/verify_science_invariants.py
    uv run python scripts/verify_rotation.py
    uv run python scripts/verify_lora2r_routing.py
    uv run python scripts/verify_corda.py
    uv run python scripts/verify_antipasto.py
    just smoke-routeA {{ ARGS }}

smoke-vanilla *ARGS:
    CUDA_VISIBLE_DEVICES="" BEARTYPE=1 {{ TRAIN }} smoke --intervention=none \
        --teacher-pool-dir={{ TEACHER_POOL }} --mix-ratio=0.5 {{ ARGS }}

smoke-routeA *ARGS:
    CUDA_VISIBLE_DEVICES="" BEARTYPE=1 {{ TRAIN }} smoke --intervention=routeA \
        --teacher-pool-dir={{ TEACHER_POOL }} --mix-ratio=0.5 \
        --eval-ablate-every=10 --eval-n-prompts=2 {{ ARGS }}

smoke-routeV *ARGS:
    CUDA_VISIBLE_DEVICES="" BEARTYPE=1 {{ TRAIN }} smoke --intervention=routeV \
        --teacher-pool-dir={{ TEACHER_POOL }} --mix-ratio=0.5 \
        --eval-ablate-every=10 --eval-n-prompts=2 {{ ARGS }}

smoke-absorb *ARGS:
    CUDA_VISIBLE_DEVICES="" BEARTYPE=1 {{ TRAIN }} smoke --intervention=absorb \
        --teacher-pool-dir={{ TEACHER_POOL }} --mix-ratio=0.5 \
        --eval-ablate-every=10 --eval-n-prompts=2 {{ ARGS }}

smoke-scorda *ARGS:
    uv run python scripts/verify_scorda.py
    CUDA_VISIBLE_DEVICES="" BEARTYPE=1 {{ TRAIN }} smoke --intervention=absorb --adapter=scorda \
        --teacher-pool-dir={{ TEACHER_POOL }} --mix-ratio=0.5 \
        --eval-ablate-every=10 --eval-n-prompts=2 {{ ARGS }}

smoke-all:
    just smoke-vanilla
    just smoke-routeA
    just smoke-routeV
    just smoke-absorb

download-tiny:
    uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('llamafactory/tiny-random-qwen3')"
