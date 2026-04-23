# hnswlib-to-go

[![CI](https://github.com/sunhailin-Leo/hnswlib-to-go/actions/workflows/ci.yml/badge.svg)](https://github.com/sunhailin-Leo/hnswlib-to-go/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/sunhailin-Leo/hnswlib-to-go.svg)](https://pkg.go.dev/github.com/sunhailin-Leo/hnswlib-to-go)

Go bindings for [hnswlib](https://github.com/nmslib/hnswlib) — a fast approximate nearest neighbor search library based on [Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320).

**hnswlib compatibility: synced with [hnswlib v0.9.0](https://github.com/nmslib/hnswlib/releases/tag/v0.9.0) via git submodule.**

## Requirements

- **Go** 1.21+
- **C++ compiler** with C++11 support (g++ or clang++)
- **Make**

## Installation

```bash
go get github.com/sunhailin-Leo/hnswlib-to-go
```

Before building your Go program, the C++ static library must be compiled:

```bash
cd $GOPATH/pkg/mod/github.com/sunhailin-Leo/hnswlib-to-go@<version>
make build
```

Or clone and build from source:

```bash
git clone --recurse-submodules https://github.com/sunhailin-Leo/hnswlib-to-go.git
cd hnswlib-to-go
make build
```

## Quick Start

```go
package main

import (
	"fmt"
	hnswgo "github.com/sunhailin-Leo/hnswlib-to-go"
)

func main() {
	// Create a new index
	//   dim=128, M=16, efConstruction=200, randomSeed=42, maxElements=10000
	index := hnswgo.New(128, 16, 200, 42, 10000, hnswgo.SpaceL2)
	defer index.Free()

	// Set search-time ef parameter
	index.SetEf(50)

	// Add vectors
	vector := make([]float32, 128)
	for i := range vector {
		vector[i] = float32(i) * 0.01
	}
	index.AddPoint(vector, 0)

	// Search for nearest neighbors
	labels, distances := index.SearchKNN(vector, 5)
	fmt.Println("Labels:", labels)
	fmt.Println("Distances:", distances)

	// Save and load
	index.Save("/tmp/my_index.bin")
	loaded := hnswgo.Load("/tmp/my_index.bin", 128, hnswgo.SpaceL2)
	defer loaded.Free()
}
```

### More Examples

| Example | What it covers |
|---------|---------------|
| [`example/basic`](example/basic/main.go) | Create → Add → Search → Save → Load → Free |
| [`example/batch`](example/batch/main.go) | `AddBatchPoints` and `SearchBatchKNN` with goroutines |
| [`example/delete_update`](example/delete_update/main.go) | `MarkDelete`, `UnmarkDelete`, `UpdatePoint`, `UpdateBatchPoints`, `GetVectorByLabel` |
| [`example/cosine_replace`](example/cosine_replace/main.go) | Cosine space, `SetNormalize`, `NewWithReplaceDeleted`, `AddPointWithReplace`, `ResizeIndex` |

## API Reference

### Index Creation

| Function | Description |
|----------|-------------|
| `New(dim, M, efConstruction, randSeed, maxElements, spaceType)` | Create a new HNSW index |
| `NewWithReplaceDeleted(dim, M, efConstruction, randSeed, maxElements, spaceType)` | Create index with replace-deleted support |
| `Load(location, dim, spaceType)` | Load index from file |

### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `dim` | `int` | Vector dimension |
| `M` | `int` | Max connections per layer (see [ALGO_PARAMS.md](https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md)) |
| `efConstruction` | `int` | Construction-time ef parameter (see [ALGO_PARAMS.md](https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md)) |
| `randSeed` | `int` | Random seed |
| `maxElements` | `uint32` | Maximum number of elements |
| `spaceType` | `string` | Distance metric (`"l2"`, `"ip"`, or `"cosine"`) |

### Distance Metrics

| Constant | Value | Description |
|----------|-------|-------------|
| `SpaceL2` | `"l2"` | Euclidean (L2) distance |
| `SpaceIP` | `"ip"` | Inner product distance |
| `SpaceCosine` | `"cosine"` | Cosine similarity (auto-normalizes vectors) |

### Data Operations

| Method | Description |
|--------|-------------|
| `AddPoint(vector, label)` | Add a single vector |
| `AddPointWithReplace(vector, label)` | Add vector, reusing deleted slots |
| `AddBatchPoints(vectors, labels, coroutines)` | Add vectors concurrently |
| `SearchKNN(vector, N)` | Search for N nearest neighbors |
| `SearchBatchKNN(vectors, N, coroutines)` | Batch search concurrently |
| `GetVectorByLabel(label)` | Retrieve stored vector by label |

### Index Management

| Method | Description |
|--------|-------------|
| `Save(location)` | Persist index to file |
| `Free()` | Release index memory |
| `SetEf(ef)` | Set search-time ef parameter |
| `SetNormalize(bool)` | Enable/disable vector normalization |
| `ResizeIndex(newMaxElements)` | Resize index capacity |

### Delete & Update

| Method | Description |
|--------|-------------|
| `MarkDelete(label)` | Soft-delete an element |
| `UnmarkDelete(label)` | Restore a soft-deleted element |
| `GetLabelIsMarkedDeleted(label)` | Check if element is deleted |
| `UpdatePoint(vector, label, prob)` | Update vector for existing label |
| `UpdateBatchPoints(vectors, labels, probs, coroutines)` | Batch update concurrently |

### Index Info

| Method | Description |
|--------|-------------|
| `GetMaxElements()` | Maximum capacity |
| `GetCurrentElementCount()` | Current number of elements |
| `GetDeleteCount()` | Number of soft-deleted elements |

## Build Targets

```bash
make build              # Build C++ library and Go package
make opt                # Build with -O3 and -march=native
make portable           # Build without -march=native (CI-friendly)
make test               # Run unit tests
make bench              # Run benchmarks
make clean              # Remove build artifacts
make help               # Show all available targets
```

### Cross-Platform Builds

Requires appropriate cross-compilation toolchains:

```bash
make build-linux-amd64   # Build for Linux x86_64
make build-linux-arm64   # Build for Linux aarch64
make build-darwin-amd64  # Build for macOS x86_64
make build-darwin-arm64  # Build for macOS ARM64
make build-windows-amd64 # Build for Windows x86_64 (MinGW)
```

### Windows Support

Windows builds require [MSYS2](https://www.msys2.org/) with MinGW-w64:

```bash
# Install MSYS2, then in MINGW64 shell:
pacman -S mingw-w64-x86_64-gcc make
make build
```

## Mutation Testing

We use [`go-mutesting`](https://github.com/avito-tech/go-mutesting) to keep the
test suite honest: the tool mutates `hnsw.go` (e.g. flips `+` to `-`, swaps
`i*b` for `i/b`, negates `if` conditions) and re-runs the suite against every
mutant. A surviving mutant points at a gap in test coverage — or an equivalent
mutant that should be documented in the PR.

```bash
# Install once
go install github.com/avito-tech/go-mutesting/cmd/go-mutesting@latest

# Full run with the shared config (scope: hnsw.go only)
make mutation

# Quick ad-hoc run against hnsw.go (no config file)
make mutation-quick
```

Scope is locked down in `.go-mutesting.yml`:

- **Mutated**: all `*.go` files in the package root (today that is just
  `hnsw.go`; new files are picked up automatically)
- **Excluded**: `third_party/**` (upstream hnswlib), `example/**`,
  `**/*_test.go`, `hnsw_wrapper.*` (C/C++ sources)
- **Operators enabled**: `arithmetic/base`, `branch/case`, `branch/if`,
  `expression/remove`, `numbers/incrementer`, `statement/remove`
- **Test runtime**: `go test -race -short -run=^TestHNSW_`. The `-race` flag
  is enabled so that mutants affecting goroutine synchronisation in the
  batch APIs show up as hard failures instead of silent survivors.

> ℹ️  Because every mutant triggers a CGO rebuild **and** a `-race` test
> binary, a full run currently takes several hours on commodity hardware.
> Prefer running it locally as a nightly / pre-release gate rather than on
> every PR.

CI runs the mutation job on Linux / Go 1.23 with `continue-on-error: true`
until the baseline mutation score stabilizes at **≥ 70 %**. See
[AGENTS.md § 7](AGENTS.md) for the full workflow and triage rules.

## Benchmarks

Measured on Apple M3 Pro, Go 1.23, `-O3 -march=native`, dim=128, 5000 indexed vectors:

| Benchmark | ns/op | B/op | allocs/op |
|-----------|------:|-----:|----------:|
| AddPoint (L2) | 1,758,072 | 0 | 0 |
| AddPoint (Cosine) | 1,728,766 | 0 | 0 |
| AddBatchPoints (1000×4 goroutines) | 2,375,182,522 | 530 | 9 |
| SearchKNN (L2, top-10) | 117,636 | 96 | 2 |
| SearchKNN (Cosine, top-10) | 86,193 | 96 | 2 |
| SearchBatchKNN (100×4 goroutines) | 3,184,033 | 16,129 | 219 |
| SaveLoad (5000 vectors) | 16,613,849 | 51 | 1 |

Run benchmarks locally:

```bash
make opt    # Build with -O3 -march=native
make bench  # Run all benchmarks
```

## Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed release notes.

- **v1.1.0** — Synced hnswlib to latest master; performance optimizations; Windows support; comprehensive tests & benchmarks; GitHub Actions CI (Go 1.21–1.26)
- **v1.0.4** — Added `UpdatePoint`, `UpdateBatchPoints`
- **v1.0.3** — Added `GetMaxElements`, `GetCurrentElementCount`, `GetDeleteCount`, `GetVectorByLabel`
- **v1.0.2** — Updated hnswlib to 0.7.0; added batch operations, delete/unmark, resize
- **v1.0.1** — Code formatting; experimental `Unload` API
- **v1.0.0** — Initial release (hnswlib 0.5.2)

## License

MIT — see [LICENSE](LICENSE) for details.
