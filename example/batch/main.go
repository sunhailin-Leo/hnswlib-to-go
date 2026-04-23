// Package main demonstrates batch operations with concurrent goroutines:
// AddBatchPoints and SearchBatchKNN.
package main

import (
	"fmt"
	"math/rand"
	"time"

	hnswgo "github.com/sunhailin-Leo/hnswlib-to-go"
)

func randomVector(dim int) []float32 {
	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = rand.Float32()
	}
	return vec
}

func main() {
	const (
		dim            = 128
		m              = 16
		efConstruction = 200
		randomSeed     = 42
		maxElements    = 10000
		numVectors     = 5000
		numQueries     = 100
		topK           = 10
		coroutines     = 4
	)

	index := hnswgo.New(dim, m, efConstruction, randomSeed, uint32(maxElements), hnswgo.SpaceL2)
	defer index.Free()
	index.SetEf(50)

	// --- 1. Batch add points ---
	vectors := make([][]float32, numVectors)
	labels := make([]uint32, numVectors)
	for i := 0; i < numVectors; i++ {
		vectors[i] = randomVector(dim)
		labels[i] = uint32(i)
	}

	start := time.Now()
	if !index.AddBatchPoints(vectors, labels, coroutines) {
		fmt.Println("AddBatchPoints failed")
		return
	}
	fmt.Printf("Added %d points in %v (coroutines=%d)\n",
		numVectors, time.Since(start), coroutines)

	// --- 2. Batch search ---
	queries := make([][]float32, numQueries)
	for i := range queries {
		queries[i] = randomVector(dim)
	}

	start = time.Now()
	labelList, distList := index.SearchBatchKNN(queries, topK, coroutines)
	fmt.Printf("Searched %d queries in %v (coroutines=%d)\n",
		numQueries, time.Since(start), coroutines)

	// Print first 3 query results
	for i := 0; i < 3 && i < len(labelList); i++ {
		fmt.Printf("\nQuery %d results:\n", i)
		for j := range labelList[i] {
			fmt.Printf("  label=%d  distance=%.6f\n", labelList[i][j], distList[i][j])
		}
	}

	fmt.Println("\nDone!")
}
