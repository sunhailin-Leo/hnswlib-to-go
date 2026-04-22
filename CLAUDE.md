# CLAUDE.md — Claude Code Configuration for hnswlib-to-go

> **Full development workflow and project conventions are defined in [AGENTS.md](AGENTS.md).** Read it first. This file contains Claude-specific extensions only.

## Quick Context

This is a Go + CGO project providing bindings to the [hnswlib](https://github.com/nmslib/hnswlib) C++ library for approximate nearest neighbor search. The C++ dependency lives in `third_party/hnswlib/` as a git submodule pinned to `v0.9.0`.

## Build & Test Commands

```bash
# Build (must run before tests)
make portable        # CI-safe build (no -march=native)
make build           # default build
make opt             # optimized (-O3 -march=native)

# Test
make test            # unit tests
make bench           # benchmarks

# Fuzz (run individual fuzz targets)
go test -fuzz=FuzzAddPointAndSearch -fuzztime=30s
go test -fuzz=FuzzSaveLoadRoundTrip -fuzztime=30s
go test -fuzz=FuzzLifecycle -fuzztime=30s
go test -fuzz=FuzzEdgeValues -fuzztime=30s

# Clean
make clean
```

## Key Files

| File | Purpose |
|------|---------|
| `hnsw.go` | Public Go API — all CGO bindings |
| `hnsw_wrapper.h` / `.cc` | C bridge between Go and C++ |
| `third_party/hnswlib/hnswlib/*.h` | Upstream C++ headers (submodule, do NOT edit) |
| `hnsw_test.go` | Unit tests |
| `hnsw_fuzz_test.go` | Fuzz tests |
| `hnsw_benchmark_test.go` | Benchmarks |
| `Makefile` | Build system |

## Rules for Claude

### Do

- **Always follow the 7-stage workflow** in AGENTS.md: Requirements → Feasibility → Code → Unit Test → Fuzz Test → Benchmark → Release.
- **Build before testing**: Run `make portable` (or `make build`) before `make test` — the C++ static library must be compiled first.
- **Test all three space types**: L2, IP, Cosine — when adding or modifying any vector operation.
- **Guard CGO calls**: Every public method must check `h.index == nil` before calling C functions.
- **Use fixed-width C types**: `uint64_t` (not `unsigned long`) for Windows LLP64 compatibility.
- **Run fuzz tests** after any CGO boundary change to catch panics/crashes with malformed inputs.

### Do Not

- **Do NOT edit** files under `third_party/hnswlib/` — they are managed by the upstream submodule.
- **Do NOT use `-march=native`** in CI or portable builds — use `make portable`.
- **Do NOT let C++ exceptions cross the CGO boundary** — catch in C++ wrapper, return error values.
- **Do NOT add Go dependencies** — this project has zero Go dependencies by design (`go.mod` has no `require` block).
- **Do NOT skip the feasibility analysis** when upstream hnswlib changes are involved.

### Style

- Go: `gofmt`, doc comments on all exported symbols, error handling via return values (not panics).
- C++: C++11, `-std=c++11`, minimize includes, `extern "C"` for all bridge functions.
- Tests: Table-driven where applicable, use `t.Cleanup()` for resource management, use `t.Run()` for subtests.
