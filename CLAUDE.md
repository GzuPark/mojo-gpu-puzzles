# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

The Modular Platform is a unified platform for AI development and deployment that includes:

- **MAX**: High-performance inference server with OpenAI-compatible endpoints for LLMs and AI models
- **Mojo**: A new programming language that bridges Python and systems programming, optimized for AI workloads

## Essential Build Commands

### Pixi Environment Management

Many directories include `pixi.toml` files for environment management. Use Pixi
when present:

```bash
# Install Pixi environment (run once per directory)
pixi install

# Run Mojo files through Pixi
pixi run mojo [file.mojo]

# Format Mojo code
pixi run format
```

## High-Level Architecture

### Repository Structure

```text
mojo-gpu-puzzles/
├── problems/                # Puzzle problems (p01-p24)
│   ├── p01/                 # Individual puzzle directories
│   ├── p02/                 # containing problem setup
│   └── ...                  # and starter code
├── solutions/               # Solution implementations
│   ├── p01/                 # Solutions for each puzzle
│   ├── p02/                 # with helper scripts
│   ├── ...                  # (run.sh, sanitizer.sh)
│   └── run.sh               # Global solution runner
├── practice/                # Practice problems
│   ├── p13/                 # Selected problems for practice
│   └── p14/                 # before tackling full solutions
├── book/                    # Documentation (mdBook)
│   ├── src/                 # Source files for documentation
│   │   ├── puzzle_*/        # Individual puzzle explanations
│   │   ├── puzzles_images/  # Visual assets
│   │   └── puzzles_animations/ # Animated explanations
│   ├── theme/               # Custom book theme
│   └── book.toml            # mdBook configuration
├── .github/                 # GitHub configuration
│   ├── workflows/           # CI/CD workflows
│   └── ISSUE_TEMPLATE/      # Issue templates
├── pixi.toml                # Pixi environment configuration
├── pyproject.toml           # Python project configuration
└── README.md                # Project overview and setup
```

### Key Architectural Patterns

1. **Language Separation**:
   - Low-level performance kernels in Mojo (`max/kernels/`)
   - High-level orchestration in Python (`max/serve/`, `max/pipelines/`)

2. **Hardware Abstraction**:
   - Platform-specific optimizations via dispatch tables
   - Support for NVIDIA/AMD GPUs, Intel/Apple CPUs
   - Device-agnostic APIs with hardware-specific implementations

3. **Memory Management**:
   - Device contexts for GPU memory management
   - Host/Device buffer abstractions
   - Careful lifetime management in Mojo code

4. **Testing Philosophy**:
   - Tests mirror source structure
   - Use `lit` tool with FileCheck validation
   - Hardware-specific test configurations
   - Migrating to `testing` module assertions

## Development Workflow

### Code Style

- Use `mojo format` for Mojo code
- Follow existing patterns in the codebase
- Add docstrings to public APIs
- Sign commits with `git commit -s`

## Critical Development Notes

### Mojo Development

- Use nightly Mojo builds for development
- Install nightly VS Code extension
- Avoid deprecated types like `Tensor` (use modern alternatives)
- Follow value semantics and ownership conventions
- Use `Reference` types with explicit lifetimes in APIs

### MAX Kernel Development

- Fine-grained control over memory layout and parallelism
- Hardware-specific optimizations (tensor cores, SIMD)
- Vendor library integration when beneficial
- Performance improvements must include benchmarks

### Common Pitfalls

- Always check Mojo function return values for errors
- Ensure coalesced memory access patterns on GPU
- Minimize CPU-GPU synchronization points
- Avoid global state in kernels
- Never commit secrets or large binary files

### Environment Variables

Many benchmarks and tests use environment variables:

- `env_get_int[param_name]=value`
- `env_get_bool[flag_name]=true/false`
- `env_get_dtype[type]=float16/float32`

## Contributing Areas

Currently accepting contributions for:

- Mojo standard library (`/mojo/stdlib/`)
- MAX AI kernels (`/max/kernels/`)

Other areas are not open for external contributions.

## Platform Support

- Linux: x86_64, aarch64
- macOS: ARM64 (Apple Silicon)
- Windows: Not currently supported

## LLM-friendly Documentation

- Docs index: <https://docs.modular.com/llms.txt>
- Mojo API docs: <https://docs.modular.com/llms-mojo.txt>
- Python API docs: <https://docs.modular.com/llms-python.txt>
- Comprehensive docs: <https://docs.modular.com/llms-full.txt>
