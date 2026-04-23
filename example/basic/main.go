// Package main demonstrates the basic HNSW index lifecycle:
// create → add points → search → save → load → free.
package main

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"

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
		maxElements    = 1000
		topK           = 5
	)

	// --- 1. Create a new L2 index ---
	index := hnswgo.New(dim, m, efConstruction, randomSeed, uint32(maxElements), hnswgo.SpaceL2)
	defer index.Free()

	// Set search-time ef (higher = more accurate, slower)
	index.SetEf(50)

	// --- 2. Add points one by one ---
	for i := 0; i < 100; i++ {
		vec := randomVector(dim)
		if !index.AddPoint(vec, uint32(i)) {
			fmt.Printf("Failed to add point %d\n", i)
		}
	}
	fmt.Printf("Added %d points (max capacity: %d)\n",
		index.GetCurrentElementCount(), index.GetMaxElements())

	// --- 3. Search for nearest neighbors ---
	query := randomVector(dim)
	labels, distances := index.SearchKNN(query, topK)
	fmt.Printf("\nSearch results (top-%d):\n", topK)
	for i := range labels {
		fmt.Printf("  label=%d  distance=%.6f\n", labels[i], distances[i])
	}

	// --- 4. Save to file ---
	tmpDir := os.TempDir()
	indexPath := filepath.Join(tmpDir, "basic_example.bin")
	if index.Save(indexPath) {
		fmt.Printf("\nIndex saved to %s\n", indexPath)
	}

	// --- 5. Load from file ---
	loaded := hnswgo.Load(indexPath, dim, hnswgo.SpaceL2)
	defer loaded.Free()

	loaded.SetEf(50)
	labels2, distances2 := loaded.SearchKNN(query, topK)
	fmt.Printf("\nLoaded index search results (top-%d):\n", topK)
	for i := range labels2 {
		fmt.Printf("  label=%d  distance=%.6f\n", labels2[i], distances2[i])
	}

	// Clean up temp file
	_ = os.Remove(indexPath)
	fmt.Println("\nDone!")
}
