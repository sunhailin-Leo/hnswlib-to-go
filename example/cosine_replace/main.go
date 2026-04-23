// Package main demonstrates Cosine space with automatic normalization,
// replace-deleted index mode, and dynamic index resizing.
package main

import (
	"fmt"
	"math/rand"

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
		dim            = 64
		m              = 16
		efConstruction = 200
		randomSeed     = 42
		initialMax     = 50
		topK           = 5
	)

	// --- 1. Cosine space (auto-normalizes vectors on insert & search) ---
	fmt.Println("=== Cosine Space ===")
	cosIndex := hnswgo.New(dim, m, efConstruction, randomSeed, uint32(initialMax), hnswgo.SpaceCosine)
	defer cosIndex.Free()
	cosIndex.SetEf(50)

	for i := 0; i < 20; i++ {
		cosIndex.AddPoint(randomVector(dim), uint32(i))
	}

	labels, distances := cosIndex.SearchKNN(randomVector(dim), topK)
	fmt.Printf("Cosine search (top-%d):\n", topK)
	for i := range labels {
		fmt.Printf("  label=%d  cosine_distance=%.6f\n", labels[i], distances[i])
	}

	// You can toggle normalization on/off at runtime
	cosIndex.SetNormalize(false)
	fmt.Println("\nNormalization disabled — raw inner-product distances will be used.")
	cosIndex.SetNormalize(true)
	fmt.Println("Normalization re-enabled.")

	// --- 2. Replace-deleted mode ---
	fmt.Println("\n=== Replace-Deleted Mode ===")
	replaceIndex := hnswgo.NewWithReplaceDeleted(
		dim, m, efConstruction, randomSeed, uint32(initialMax), hnswgo.SpaceL2,
	)
	defer replaceIndex.Free()
	replaceIndex.SetEf(50)

	// Add some points
	for i := 0; i < 10; i++ {
		replaceIndex.AddPoint(randomVector(dim), uint32(i))
	}
	fmt.Printf("Elements: %d\n", replaceIndex.GetCurrentElementCount())

	// Soft-delete label 3, then reuse its slot with AddPointWithReplace
	replaceIndex.MarkDelete(3)
	fmt.Printf("After MarkDelete(3): deleted=%d\n", replaceIndex.GetDeleteCount())

	replaceIndex.AddPointWithReplace(randomVector(dim), 100)
	fmt.Printf("After AddPointWithReplace(label=100): elements=%d, deleted=%d\n",
		replaceIndex.GetCurrentElementCount(), replaceIndex.GetDeleteCount())

	// --- 3. Dynamic resizing ---
	fmt.Println("\n=== Dynamic Resizing ===")
	fmt.Printf("Before resize: maxElements=%d\n", replaceIndex.GetMaxElements())

	if replaceIndex.ResizeIndex(200) {
		fmt.Printf("After resize:  maxElements=%d\n", replaceIndex.GetMaxElements())
	}

	// Now we can add more points beyond the original capacity
	for i := 10; i < 30; i++ {
		replaceIndex.AddPoint(randomVector(dim), uint32(i))
	}
	fmt.Printf("Elements after adding more: %d\n", replaceIndex.GetCurrentElementCount())

	fmt.Println("\nDone!")
}
