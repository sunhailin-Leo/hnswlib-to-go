# hnswlib-to-go
Hnswlib to go. Golang interface to hnswlib(https://github.com/nmslib/hnswlib). This is a golang interface of [hnswlib](https://github.com/nmslib/hnswlib). For more information, please follow [hnswlib](https://github.com/nmslib/hnswlib) and [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs.](https://arxiv.org/abs/1603.09320).

**But in this project, we make compatible hnswlib to 0.7.0.**


### Version

* version 1.0.3
  * Add `GetMaxElements`, `GetCurrentElementCount`, `GetDeleteCount`, `GetVectorByLabel` APIs

* version 1.0.2
  * Update hnswlib compatible version to 0.7.0
  * Add `AddBatchPoints`, `SearchBatchKNN`, `SetNormalize`, `ResizeIndex`, `MarkDelete`, `UnmarkDelete`, `GetLabelIsMarkedDeleted` API

* version 1.0.1
  * Code format
  * Add an api support unload the graph(Experimental)

* version 1.0.0
  * hnswlib compatible version 0.5.2.


### Build

* Linux/MacOS
  * Build Golang Env
  * `go mod init`
  * `make`

### Usage

* When building golang program, please add `export CGO_CXXFLAGS=-std=c++11` command before `go build / run / test ...`

| argument       | type | |
| -------------- | ---- | ----- |
| dim            | int  | vector dimension |
| M              | int  | see[ALGO_PARAMS.md](https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md) |
| efConstruction | int  | see[ALGO_PARAMS.md](https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md) |
| randomSeed     | int  | random seed for hnsw |
| maxElements    | int  | max records in data |
| spaceType      | str  | |

| spaceType | distance          |
| --------- |:-----------------:|
| ip        | inner product     |
| cosine    | cosine similarity |
| l2        | l2                |
