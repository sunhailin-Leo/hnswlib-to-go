// Package main demonstrates soft-delete, undelete, point update, and
// vector retrieval APIs.
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
		dim            = 8
		m              = 16
		efConstruction = 200
		randomSeed     = 42
		maxElements    = 100
	)

	index := hnswgo.New(dim, m, efConstruction, randomSeed, uint32(maxElements), hnswgo.SpaceL2)
	defer index.Free()
	index.SetEf(50)

	// Add 10 points
	for i := 0; i < 10; i++ {
		index.AddPoint(randomVector(dim), uint32(i))
	}
	fmt.Printf("Elements: %d, Deleted: %d\n",
		index.GetCurrentElementCount(), index.GetDeleteCount())

	// --- 1. Soft-delete a point ---
	targetLabel := uint32(5)
	index.MarkDelete(targetLabel)
	fmt.Printf("\nAfter MarkDelete(%d):\n", targetLabel)
	fmt.Printf("  IsDeleted: %v\n", index.GetLabelIsMarkedDeleted(targetLabel))
	fmt.Printf("  DeleteCount: %d\n", index.GetDeleteCount())

	// Deleted points are excluded from search results
	labels, _ := index.SearchKNN(randomVector(dim), 10)
	fmt.Printf("  Search returns %d labels (label %d excluded)\n", len(labels), targetLabel)

	// --- 2. Restore a deleted point ---
	index.UnmarkDelete(targetLabel)
	fmt.Printf("\nAfter UnmarkDelete(%d):\n", targetLabel)
	fmt.Printf("  IsDeleted: %v\n", index.GetLabelIsMarkedDeleted(targetLabel))
	fmt.Printf("  DeleteCount: %d\n", index.GetDeleteCount())

	// --- 3. Update a single point ---
	newVector := make([]float32, dim)
	for i := range newVector {
		newVector[i] = float32(i) * 0.1
	}
	updateSuccess := index.UpdatePoint(newVector, targetLabel, 1.0)
	fmt.Printf("\nUpdatePoint(label=%d): success=%v\n", targetLabel, updateSuccess)

	// Verify the update via GetVectorByLabel
	retrieved := index.GetVectorByLabel(targetLabel)
	fmt.Printf("  Retrieved vector: %v\n", retrieved)

	// --- 4. Batch update multiple points ---
	updateVectors := make([][]float32, 3)
	updateLabels := []uint32{0, 1, 2}
	updateProbs := []float32{1.0, 1.0, 1.0}
	for i := range updateVectors {
		updateVectors[i] = randomVector(dim)
	}

	batchSuccess := index.UpdateBatchPoints(updateVectors, updateLabels, updateProbs, 2)
	fmt.Printf("\nUpdateBatchPoints(labels=%v, coroutines=2): success=%v\n",
		updateLabels, batchSuccess)

	// Verify one of the batch-updated points
	retrieved = index.GetVectorByLabel(0)
	fmt.Printf("  Retrieved label=0 vector: %v\n", retrieved)

	fmt.Println("\nDone!")
}
