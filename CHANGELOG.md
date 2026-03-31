# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
