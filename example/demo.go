package main

import (
	"fmt"
	"math/rand"
	"time"

	hnswgo "github.com/sunhailin-Leo/hnswlib-to-go"
)

func randVector(dim int) []float32 {
	vec := make([]float32, dim)
	for j := 0; j < dim; j++ {
		vec[j] = rand.Float32()
	}
	return vec
}

func main() {
	var dim, M, ef int = 128, 32, 300
	// Max elements
	var maxElements uint32 = 1000
	// Distance cosine
	var spaceType, indexLocation string = "cosine", "hnsw_demo_index.bin"
	var randomSeed int = 100
	// Init new index
	h := hnswgo.New(dim, M, ef, randomSeed, maxElements, spaceType)
	// Insert 1000 vectors to index. Label Type is uint32
	var i uint32
	for ; i < maxElements; i++ {
		h.AddPoint(randVector(dim), i)
	}
	h.Save(indexLocation)
	h = hnswgo.Load(indexLocation, dim, spaceType)
	// Search vector with maximum 5 NN
	h.SetEf(15)
	searchVector := randVector(dim)
	// Count query time
	startTime := time.Now().UnixNano()
	labels, vectors := h.SearchKNN(searchVector, 5)
	endTime := time.Now().UnixNano()
	fmt.Println(endTime - startTime)
	fmt.Println(labels, vectors)
}
