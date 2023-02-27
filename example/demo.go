package main

import (
	"fmt"
	"math/rand"
	"runtime"
	"time"

	hnswgo "github.com/sunhailin-Leo/hnswlib-to-go"
)

func toMegaBytes(bytes uint64) float64 {
	return float64(bytes) / 1024 / 1024
}

func traceMemStats() {
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	var result = make([]float64, 8)
	result[0] = float64(ms.HeapObjects)
	result[1] = toMegaBytes(ms.HeapAlloc)
	result[2] = toMegaBytes(ms.TotalAlloc)
	result[3] = toMegaBytes(ms.HeapSys)
	result[4] = toMegaBytes(ms.HeapIdle)
	result[5] = toMegaBytes(ms.HeapReleased)
	result[6] = toMegaBytes(ms.HeapIdle - ms.HeapReleased)
	result[7] = toMegaBytes(ms.Alloc)

	fmt.Printf("%d\t", time.Now().Unix())
	for _, v := range result {
		fmt.Printf("%.2f\t", v)
	}
	fmt.Printf("\n")
	time.Sleep(2 * time.Second)
}

func randVector(dim int) []float32 {
	vec := make([]float32, dim)
	for j := 0; j < dim; j++ {
		vec[j] = rand.Float32()
	}
	return vec
}

// 单个写入
func exampleAddPoint(indexFileName string) {
	var dim, M, ef = 128, 32, 300
	// 最大的 elements 数
	var maxElements uint32 = 10000
	// 定义距离 cosine
	var spaceType = "cosine"
	var randomSeed = 100
	fmt.Println("Before Create HNSW")
	traceMemStats()
	// Init new index
	h := hnswgo.New(dim, M, ef, randomSeed, maxElements, spaceType)
	// Insert 1000 vectors to index. Label Type is uint32
	var i uint32
	for ; i < maxElements; i++ {
		if i%1000 == 0 {
			fmt.Println(i)
		}
		h.AddPoint(randVector(dim), i)
	}
	h.Save(indexFileName)
}

// 批量写入
func exampleBatchAddPoint(indexFileName string) {
	var dim, M, ef = 128, 32, 300

	// 最大的 elements 数
	var maxElements uint32 = 20000

	// 定义距离 cosine
	var spaceType = "cosine"
	var randomSeed = 100
	fmt.Println("Before Create HNSW")

	// 初始化 Init new index
	h := hnswgo.New(dim, M, ef, randomSeed, maxElements, spaceType)

	vectorList := make([][]float32, maxElements)
	ids := make([]uint32, maxElements)
	var i uint32
	for ; i < maxElements; i++ {
		if i%1000 == 0 {
			fmt.Println(i)
		}
		vectorList[i] = randVector(dim)
		ids[i] = i
	}
	h.AddBatchPoints(vectorList, ids, 10)

	// 保存索引 Save Index
	h.Save(indexFileName)
}

// 读取
func exampleLoadIndex(indexFileName, spaceType string, dim int) {
	h := hnswgo.Load(indexFileName, dim, spaceType)
	// Search vector with maximum 5 NN
	h.SetEf(15)
	searchVector := randVector(dim)
	// Count query time
	startTime := time.Now().UnixNano()
	labels, vectors := h.SearchKNN(searchVector, 5)
	endTime := time.Now().UnixNano()
	fmt.Println(endTime - startTime)
	fmt.Println(labels, vectors)

	// Test ResizeIndex API
	isResize := h.ResizeIndex(12000)
	fmt.Println("Size flag: ", isResize)

	// Test Mark API
	isMarkDelete := h.MarkDelete(10)
	fmt.Println("isMarkDelete: ", isMarkDelete)

	labelIsDelete := h.GetLabelIsMarkedDeleted(10)
	fmt.Println("labelIsDelete: ", labelIsDelete)

	isUnmarkDelete := h.UnmarkDelete(10)
	fmt.Println("isUnmarkDelete: ", isUnmarkDelete)

	// Test Unload API
	fmt.Println("Before Unload")
	traceMemStats()
	h.Unload()
	fmt.Println("After Unload")
	traceMemStats()
}

func main() {
	// 单条写入 add index point by point
	exampleAddPoint("hnsw_demo_single.bin")
	// 测试读取 test loading
	exampleLoadIndex("hnsw_demo_single.bin", "cosine", 128)

	// 批量写入 add index with batch mode
	//exampleBatchAddPoint("hnsw_demo_multiple.bin")
	// 测试读取 test loading
	//exampleLoadIndex("hnsw_demo_multiple.bin", "cosine", 128)
}
