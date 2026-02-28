# Contributing to kvik-rs

Thank you for your interest in contributing to kvik-rs!

## Getting Started

1. Fork and clone the repository
2. Install prerequisites: Rust toolchain, CUDA toolkit, cuFile/GDS drivers (optional for host-only work)
3. Run `just ci` to verify your environment

## Development Workflow

```sh
just check          # Compile check
just test           # Run all tests
just clippy         # Lint
just fmt            # Format code
just ci             # Full CI suite (fmt + clippy + test + doc)
```

GPU-specific tests are `#[ignore]` by default and require a CUDA device:

```sh
just test-vram-stress   # GPU VRAM stress tests
just test-gds-memory    # GDS memory transfer validation
```

## Guidelines

- **Test first.** Write tests before implementation. Reference C++ kvikio tests where applicable.
- **No async runtime.** The library is synchronous by design. Async integration belongs in the caller.
- **Use cudarc's safe API** where possible. Drop to `cudarc::cufile::result` only when necessary, with safety comments.
- **Run `just ci`** before submitting. All checks must pass.

## Submitting Changes

1. Create a feature branch from `main`
2. Make focused commits with clear messages
3. Ensure `just ci` passes
4. Open a pull request with a description of what and why

## Reporting Issues

Open an issue on the GitHub repository with:
- What you expected vs. what happened
- Steps to reproduce
- Environment details (OS, CUDA version, GPU model, filesystem type for GDS issues)
