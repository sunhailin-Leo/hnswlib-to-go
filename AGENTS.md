# AGENTS.md — hnswlib-to-go Development Guide

This file defines the development workflow, conventions, and quality gates for **hnswlib-to-go** — Go bindings for the [hnswlib](https://github.com/nmslib/hnswlib) approximate nearest neighbor search library.

## Project Overview

- **Language**: Go (CGO bindings to C++)
- **Upstream dependency**: [nmslib/hnswlib](https://github.com/nmslib/hnswlib) — managed as a git submodule at `third_party/hnswlib`, pinned to a specific release tag (currently `v0.9.0`)
- **Build system**: Makefile + CGO
- **CI**: GitHub Actions (Linux, macOS, Windows × Go 1.21–1.26)

## Repository Structure

```
.
├── third_party/hnswlib/    # git submodule → nmslib/hnswlib (pinned tag)
│   └── hnswlib/            # upstream C++ headers
├── hnsw_wrapper.h          # C bridge header
├── hnsw_wrapper.cc         # C bridge implementation
├── hnsw.go                 # Go public API (CGO bindings)
├── hnsw_test.go            # unit tests
├── hnsw_fuzz_test.go       # fuzz tests
├── hnsw_benchmark_test.go  # benchmarks
├── example/
│   ├── basic/              # basic lifecycle (create, add, search, save, load)
│   ├── batch/              # batch add & batch search with goroutines
│   ├── delete_update/      # soft-delete, undelete, update, vector retrieval
│   └── cosine_replace/     # cosine space, replace-deleted mode, resize
├── Makefile                # build, test, bench, cross-compile targets
├── .go-mutesting.yml       # mutation testing configuration
├── .github/workflows/      # CI configuration
├── AGENTS.md               # this file (AI agent guidelines)
├── CLAUDE.md               # Claude-specific prompt extensions
├── CHANGELOG.md            # release history
└── README.md               # user-facing documentation
```

## Development Workflow

Every change — feature, bugfix, or refactor — **must** follow these stages in order:

### 1. Requirements Analysis

- Clearly define **what** the change does and **why** it is needed.
- Identify which public APIs are affected (new, modified, or deprecated).
- List any upstream hnswlib changes required.

### 2. Feasibility Analysis

- Check upstream hnswlib API compatibility (review `third_party/hnswlib/hnswlib/*.h`).
- Evaluate CGO constraints (thread safety, memory management, type mapping).
- Identify cross-platform impacts (Linux, macOS, Windows).
- Estimate breaking change risk for existing users.

### 3. Code Development

#### Conventions

- **Go code** follows standard `gofmt` formatting and [Effective Go](https://go.dev/doc/effective-go) guidelines.
- **C++ bridge code** (`hnsw_wrapper.h`, `hnsw_wrapper.cc`) uses C++11, no exceptions leaking to Go, all errors returned as values.
- All public Go functions must have doc comments.
- Use `uint32` for labels, `float32` for vectors (matching C bridge types).
- All CGO calls must guard against `nil` index pointers.

#### Build Commands

```bash
make build       # default build (C++ library + Go package)
make opt         # optimized build (-O3 -march=native)
make portable    # CI-safe build (no -march=native)
make clean       # remove build artifacts
```

### 4. Unit Test Verification

- Every public API function must have test coverage.
- Tests must cover **all three space types**: L2, IP, Cosine.
- Tests must cover nil/freed index safety.
- Tests must cover invalid input edge cases.

```bash
make test
```

**Pass criteria**: All tests pass with zero failures, no data races (`-race` flag in CI).

### 5. Fuzz Test Verification

- Fuzz tests (`hnsw_fuzz_test.go`) exercise the CGO boundary with random/malformed inputs.
- Must cover: core operations (AddPoint, SearchKNN, UpdatePoint), lifecycle (New/Free/Save/Load), edge values (NaN, Inf, zero vectors), and Save→Load round-trip consistency.

```bash
go test -fuzz=FuzzAddPointAndSearch -fuzztime=30s
go test -fuzz=FuzzSaveLoadRoundTrip -fuzztime=30s
go test -fuzz=FuzzLifecycle -fuzztime=30s
go test -fuzz=FuzzEdgeValues -fuzztime=30s
```

**Pass criteria**: No panics, no crashes, no data corruption after the specified fuzz duration.

### 6. Benchmark Verification

- Run benchmarks to detect performance regressions.
- Compare against baseline numbers in README.

```bash
make bench
```

**Pass criteria**: No significant regressions (>20% degradation) in ns/op, B/op, or allocs/op compared to the previous release.

### 7. Mutation Test Verification (Recommended)

Mutation testing evaluates **test-suite quality** by applying small code
changes (mutants) to `hnsw.go` and verifying that the existing tests detect
each change. A surviving mutant indicates a gap in test coverage or an
equivalent mutant that should be documented.

Tooling: [`go-mutesting`](https://github.com/avito-tech/go-mutesting) (avito-tech fork).

```bash
# Install once
go install github.com/avito-tech/go-mutesting/cmd/go-mutesting@latest

# Full run (uses .go-mutesting.yml, respects CGO env)
make mutation

# Quick run against hnsw.go only (no config file)
make mutation-quick
```

Scope (enforced in `.go-mutesting.yml`):

- **Mutated**: every `*.go` file in the package root
  (currently only `hnsw.go`, but new Go files are picked up automatically).
- **Excluded**: `third_party/**` (upstream), `example/**` (demo),
  `**/*_test.go`, and `hnsw_wrapper.*` (C/C++ sources).
- **Operators**: `arithmetic/base`, `branch/case`, `branch/if`,
  `expression/remove`, `numbers/incrementer`, `statement/remove`.
  `statement/remove` is intentionally enabled so that tests which let
  `C.xxx` calls pass without asserting their side-effects are flagged.
- **Runtime**: each mutant is validated with `go test -race -short -run=^TestHNSW_`.
  The race detector is essential here — most of the surface area is concurrent
  batch operations against a shared C index, so mutants that subtly break
  synchronisation (e.g. dropping a `wg.Add(1)`) otherwise slip past
  correctness-only assertions. Expect a ~3×–4× wall-clock increase per mutant
  compared to a non-`-race` baseline.

**Pass criteria**: Mutation score **≥ 70%** on `hnsw.go`. Every surviving
mutant must be triaged — either a new test case is added to kill it, or the
reason it is an equivalent mutant is recorded in the PR description.

In CI, the `mutation` job currently runs with `continue-on-error: true` so
that the score remains informational. Flip it to `false` once the baseline
is stable.

### 8. Release

1. Update `CHANGELOG.md` with all changes (follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format).
2. Update version references in `README.md` if applicable.
3. If upstream hnswlib was updated, document the new tag/commit in both `CHANGELOG.md` and `README.md`.
4. Create a git tag following [Semantic Versioning](https://semver.org/): `vMAJOR.MINOR.PATCH`.
5. Push tag to trigger CI validation on all platforms.

## Upstream Submodule Management

The hnswlib C++ library is managed as a git submodule:

```bash
# Update submodule to a new release tag
cd third_party/hnswlib
git fetch --tags
git checkout <new-tag>
cd ../..
git add third_party/hnswlib
git commit -m "chore: update hnswlib submodule to <new-tag>"
```

**Important**: After updating the submodule, always:
1. Verify API compatibility by reviewing header diffs.
2. Run the full test suite (`make test`).
3. Run fuzz tests.
4. Run benchmarks to detect performance changes.

## Cross-Platform Notes

- **Windows**: Requires MinGW-w64; CGO LDFLAGS differ from Unix.
- **macOS**: Uses `libc++` (not `libstdc++`).
- **Linux**: Uses `libstdc++`.
- All C bridge types use fixed-width integers (`uint64_t`, `int`) for LLP64/LP64 compatibility.

## Code Review Checklist

- [ ] Requirements clearly stated
- [ ] Feasibility analysis completed
- [ ] All new public APIs have doc comments
- [ ] Unit tests added/updated for all space types
- [ ] Fuzz tests cover new CGO boundary code
- [ ] Benchmarks show no significant regression
- [ ] Mutation score ≥ 70% on `hnsw.go` (or surviving mutants justified)
- [ ] CHANGELOG.md updated
- [ ] CI passes on all platforms (Linux, macOS, Windows)
