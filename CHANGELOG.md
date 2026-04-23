# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Mutation testing pipeline via [`go-mutesting`](https://github.com/avito-tech/go-mutesting):
  - New `.go-mutesting.yml` config scoped to every `*.go` source file in the
    package root (currently just `hnsw.go`; new files are picked up
    automatically). `third_party/**`, `example/**`, `**/*_test.go`, and
    `hnsw_wrapper.*` (C/C++) are excluded.
  - Enabled operators: `arithmetic/base`, `branch/case`, `branch/if`,
    `expression/remove`, `numbers/incrementer`, and `statement/remove`
    (the last one catches tests that let `C.xxx` side-effect calls pass
    without asserting their observable effects).
  - Per-mutant test command runs with `-race` so that mutants which subtly
    break goroutine synchronisation in the batch APIs surface as hard
    failures instead of silent survivors. `-timeout` and the outer
    per-mutant `timeout` were widened (60s→120s / 120s→240s) to absorb
    the race-detector slowdown.
  - New Makefile targets: `make mutation` (config-driven) and
    `make mutation-quick` (CLI-only run against `hnsw.go`).
  - New `mutation` job in `.github/workflows/ci.yml`
    (Linux / Go 1.23, `continue-on-error: true` until the baseline stabilizes).
  - AGENTS.md gains a dedicated "Mutation Test Verification" section
    and a new code-review checkbox (mutation score ≥ 70% on `hnsw.go`).
- New identity-mapping tests exercising the goroutine-splitting arithmetic of
  the batch APIs across multiple `(size, coroutines)` combinations
  (`divisible`, `indivisible`, `prime_size`, `single_coroutine`,
  `coroutines_gt_items`, `coroutines_eq_items`):
  - `TestHNSW_AddBatchPoints_IdentityMapping`
  - `TestHNSW_UpdateBatchPoints_IdentityMapping`
  - `TestHNSW_SearchBatchKNN_IdentityMapping`

### Fixed
- `UpdateBatchPoints`: fixed **two** copy-paste typos in the goroutine-splitting
  logic (both introduced when this function was originally cloned from
  `AddBatchPoints`):
  1. **Tail-remainder guard** — the sentinel check was
     `if i == coroutines+1 && len(vectors) > end` instead of
     `i == coroutines-1`. Since `i` iterates over `[0, coroutines-1]`,
     the condition was unreachable, so whenever
     `len(vectors) % coroutines != 0` the tail elements were silently
     **dropped** (e.g. with 103 vectors / 4 goroutines, indices 100–102
     were never updated).
  2. **Probability slice index** — `updateNeighborProbabilities[i/b:end]`
     instead of `[i*b:end]`, which mis-aligned probabilities with their
     vectors/labels and additionally panicked with `integer divide by zero`
     whenever `coroutines > len(vectors)` (because `b = len(vectors)/coroutines`
     was `0`).

  Both were discovered while extending mutation-testing coverage of the
  batch APIs. The second bug turned out to also mask the first — fixing
  `i/b → i*b` let go-mutesting's `coroutines+1 → coroutines-1` mutant
  surface as a FAIL, which exposed the tail-remainder regression.
- `GetVectorByLabel`: relaxed the success check from `result < 1` to
  `result < 0` so that only the documented error sentinel (negative values
  returned by the C bridge) triggers a `nil` return.
- `TestHNSW_MarkDeleteUnmarkDelete_Chain`: new chain test exercising the
  full MarkDelete → IsDeleted → UnmarkDelete → IsDeleted → Search lifecycle
  for all three space types; kills `statement/remove` mutants that drop
  `C.markDelete` / `C.unmarkDelete` calls.
- `TestHNSW_NilIndex_ExtendedMethods`: new nil-safety test covering
  `MarkDelete`, `UnmarkDelete`, `GetLabelIsMarkedDeleted`, `SetEf`,
  `SetNormalize`, `ResizeIndex`, `UpdatePoint`, `UpdateBatchPoints`,
  and `GetVectorByLabel` on a freed index.

### Changed
- `SearchBatchKNN`: each goroutine now allocates a single C-type buffer
  and reuses it across all queries in its batch, eliminating per-query
  `sync.Pool` get/put overhead. Wall-clock latency dropped **~24%**
  (4.17 ms → 3.18 ms on 100 queries × 4 goroutines).
- Benchmark accuracy: `BenchmarkAddPoint_L2` and `BenchmarkAddPoint_Cosine`
  now pre-generate vectors before `b.ResetTimer()`, so the reported B/op
  reflects the true `AddPoint` cost (0 B/op) instead of including the
  `benchRandVector` allocation (previously 512 B/op).

## [v1.1.0] - 2026-03-31

### Added
- Synced hnswlib C++ headers to latest [nmslib/hnswlib master](https://github.com/nmslib/hnswlib/tree/master/hnswlib)
- New `hnswlib/stop_condition.h` header from upstream
- `NewWithReplaceDeleted()` — create index with replace-deleted support
- `AddPointWithReplace()` — add vector reusing deleted slots
- `Free()` — explicit memory release for HNSW index
- Nil-safety checks for all public methods (return zero values instead of panicking)
- Comprehensive unit tests: 19 test functions covering all space types (L2, IP, Cosine)
- Performance benchmarks: 7 benchmark functions (AddPoint, SearchKNN, BatchOps, SaveLoad)
- Multi-platform Makefile with `portable`, `opt`, `bench`, `lint` targets
- Cross-compilation targets: `build-linux-amd64`, `build-linux-arm64`, `build-darwin-amd64`, `build-darwin-arm64`, `build-windows-amd64`
- Windows support via MinGW-w64 (conditional CGO LDFLAGS, `uint64_t` types)
- GitHub Actions CI: Linux, macOS, Windows matrix with Go 1.21–1.26

### Changed
- Upgraded hnswlib from v0.7.0 to latest master
- All C bridge types changed from `unsigned long int` to `uint64_t` for cross-platform safety (Windows LLP64 compatibility)
- `SearchKNN` now uses `sync.Pool` to reuse C-type buffers — **allocs reduced 50%** (4 → 2), **L2 search ~12% faster**
- `SearchBatchKNN` uses lock-free per-index writes instead of mutex — **allocs reduced 49%** (412 → 211)
- `normalizeVector` optimized to in-place modification (no slice return/copy)
- Removed unnecessary `runtime.GC()` call from `Free()`
- `go.mod` bumped to Go 1.21

### Fixed
- Fixed 3 upstream bugs in `hnswalg.h`: `internal_id` → `internalId` variable name
- Fixed `UpdateBatchPoints` condition bug (`&&` → `||` for parameter validation)

## [v1.0.4]

### Added
- `UpdatePoint()` — update vector for existing label
- `UpdateBatchPoints()` — batch update with concurrent goroutines

## [v1.0.3]

### Added
- `GetMaxElements()` — query maximum index capacity
- `GetCurrentElementCount()` — query current element count
- `GetDeleteCount()` — query soft-deleted element count
- `GetVectorByLabel()` — retrieve stored vector by label

## [v1.0.2]

### Changed
- Updated hnswlib to v0.7.0

### Added
- Batch operations: `AddBatchPoints`, `SearchBatchKNN`
- Soft delete: `MarkDelete`, `UnmarkDelete`, `GetLabelIsMarkedDeleted`
- `ResizeIndex()` — dynamically resize index capacity

## [v1.0.1]

### Changed
- Code formatting improvements
- Experimental `Unload` API (deprecated in v1.1.0, use `Free`)

## [v1.0.0]

### Added
- Initial release with hnswlib v0.5.2
- Core API: `New`, `Load`, `Save`, `AddPoint`, `SearchKNN`, `SetEf`
- Distance metrics: L2, Inner Product, Cosine
