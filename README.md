# hnswlib-to-go

[![CI](https://github.com/sunhailin-Leo/hnswlib-to-go/actions/workflows/ci.yml/badge.svg)](https://github.com/sunhailin-Leo/hnswlib-to-go/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/sunhailin-Leo/hnswlib-to-go.svg)](https://pkg.go.dev/github.com/sunhailin-Leo/hnswlib-to-go)

Go bindings for [hnswlib](https://github.com/nmslib/hnswlib) — a fast approximate nearest neighbor search library based on [Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320).

**hnswlib compatibility: synced with [hnswlib master](https://github.com/nmslib/hnswlib/tree/master/hnswlib).**

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

## Benchmarks

Measured on Apple M3 Pro, Go 1.23, `-O3 -march=native`, dim=128, 5000 indexed vectors:

| Benchmark | ns/op | B/op | allocs/op |
|-----------|------:|-----:|----------:|
| AddPoint (L2) | 1,805,974 | 512 | 1 |
| AddPoint (Cosine) | 1,476,631 | 512 | 1 |
| AddBatchPoints (1000×4 goroutines) | 2,949,077,675 | 454 | 9 |
| SearchKNN (L2, top-10) | 119,831 | 96 | 2 |
| SearchKNN (Cosine, top-10) | 89,467 | 96 | 2 |
| SearchBatchKNN (100×4 goroutines) | 4,231,052 | 15,650 | 211 |
| SaveLoad (5000 vectors) | 16,149,372 | 50 | 1 |

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
