package main

import (
	"fmt"
	"math/rand"
	"reflect"
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
func exampleAddPoint(indexFileName string) []float32 {
	var dim, M, ef = 128, 32, 300
	// 最大的 elements 数
	var maxElements uint32 = 100
	// 定义距离 cosine
	var spaceType = "cosine"
	var randomSeed = 2000
	fmt.Println("Before Create HNSW")
	traceMemStats()
	// Init new index
	h := hnswgo.New(dim, M, ef, randomSeed, maxElements, spaceType)

	// randomIndex to test the api GetVectorByLabel
	var randomIndex []float32

	// Insert 1000 vectors to index. Label Type is uint32
	var i uint32
	for ; i < maxElements; i++ {
		if i%1000 == 0 {
			fmt.Println(i)
		}
		randVec := randVector(dim)
		h.AddPoint(randVec, i)
		if i == 0 {
			randomIndex = randVec
		}
	}
	h.Save(indexFileName)
	return randomIndex
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
func exampleLoadIndex(indexFileName, spaceType string, dim int) []float32 {
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

	// Test GetMaxElements API Before Resize
	maxElementsBeforeResize := h.GetMaxElements()
	currentElementsBeforeResize := h.GetCurrentElementCount()
	fmt.Println("maxElements, currentElements(before resize): ", maxElementsBeforeResize, currentElementsBeforeResize)

	// Test ResizeIndex API
	isResize := h.ResizeIndex(12000)
	fmt.Println("Size flag: ", isResize)

	// Test GetMaxElements API After Resize
	maxElementsAfterResize := h.GetMaxElements()
	currentElementsAfterResize := h.GetCurrentElementCount()
	fmt.Println("maxElements, currentElements(after resize): ", maxElementsAfterResize, currentElementsAfterResize)

	// Test GetDeleteCount API
	deleteCountBeforeDelete := h.GetDeleteCount()
	fmt.Println("GetDeleteCount(before): ", deleteCountBeforeDelete)

	// Test Mark API
	isMarkDelete := h.MarkDelete(10)
	fmt.Println("isMarkDelete: ", isMarkDelete)

	labelIsDelete := h.GetLabelIsMarkedDeleted(10)
	fmt.Println("labelIsDelete: ", labelIsDelete)

	// Test GetDeleteCount API
	deleteCountBeforeAfter := h.GetDeleteCount()
	fmt.Println("GetDeleteCount(after): ", deleteCountBeforeAfter)

	isUnmarkDelete := h.UnmarkDelete(10)
	fmt.Println("isUnmarkDelete: ", isUnmarkDelete)

	// Test GetVectorByLabel API
	getVectorByIdRes := h.GetVectorByLabel(0, dim)
	fmt.Println("Vector: ", getVectorByIdRes)

	// Test Unload API
	fmt.Println("Before Unload")
	traceMemStats()
	h.Unload()
	fmt.Println("After Unload")
	traceMemStats()

	return getVectorByIdRes
}

func main() {
	// 单条写入 add index point by point
	demoVector := exampleAddPoint("hnsw_demo_single.bin")
	// 测试读取 test loading
	demoSearchVector := exampleLoadIndex("hnsw_demo_single.bin", "cosine", 128)
	// test GetVectorByLabel API
	isEqual := reflect.DeepEqual(demoVector, demoSearchVector)
	fmt.Println("GetVectorByLabel return data is equal: ", isEqual)

	// 批量写入 add index with batch mode
	exampleBatchAddPoint("hnsw_demo_multiple.bin")
	// 测试读取 test loading
	exampleLoadIndex("hnsw_demo_multiple.bin", "cosine", 128)
}
