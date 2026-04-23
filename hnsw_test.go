package hnswgo

import (
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

// randVector generates a random vector with the given dimension
func randVector(dim int) []float32 {
	vector := make([]float32, dim)
	for i := 0; i < dim; i++ {
		vector[i] = rand.Float32()
	}
	return vector
}

// TestHNSW_New tests the New function for all space types
func TestHNSW_New(t *testing.T) {
	spaceTypes := []string{SpaceL2, SpaceIP, SpaceCosine}

	for _, spaceType := range spaceTypes {
		t.Run(spaceType, func(t *testing.T) {
			hnsw := New(10, 16, 200, 42, 1000, spaceType)
			if hnsw == nil {
				t.Fatal("New returned nil")
			}
			if !hnsw.Free() {
				t.Error("Free failed")
			}
		})
	}
}

// TestHNSW_NewWithReplaceDeleted tests the NewWithReplaceDeleted function
func TestHNSW_NewWithReplaceDeleted(t *testing.T) {
	spaceTypes := []string{SpaceL2, SpaceIP, SpaceCosine}

	for _, spaceType := range spaceTypes {
		t.Run(spaceType, func(t *testing.T) {
			hnsw := NewWithReplaceDeleted(10, 16, 200, 42, 1000, spaceType)
			if hnsw == nil {
				t.Fatal("NewWithReplaceDeleted returned nil")
			}
			if !hnsw.Free() {
				t.Error("Free failed")
			}
		})
	}
}

// TestHNSW_AddPoint tests adding points to the index
func TestHNSW_AddPoint(t *testing.T) {
	spaceTypes := []string{SpaceL2, SpaceIP, SpaceCosine}

	for _, spaceType := range spaceTypes {
		t.Run(spaceType, func(t *testing.T) {
			hnsw := New(10, 16, 200, 42, 1000, spaceType)
			t.Cleanup(func() { hnsw.Free() })

			vector := randVector(10)
			if !hnsw.AddPoint(vector, 1) {
				t.Error("AddPoint failed")
			}

			if hnsw.GetCurrentElementCount() != 1 {
				t.Errorf("Expected 1 element, got %d", hnsw.GetCurrentElementCount())
			}
		})
	}
}

// TestHNSW_SearchKNN tests searching for nearest neighbors
func TestHNSW_SearchKNN(t *testing.T) {
	spaceTypes := []string{SpaceL2, SpaceIP, SpaceCosine}

	for _, spaceType := range spaceTypes {
		t.Run(spaceType, func(t *testing.T) {
			hnsw := New(10, 16, 200, 42, 1000, spaceType)
			t.Cleanup(func() { hnsw.Free() })

			// Add some points
			for i := uint32(0); i < 10; i++ {
				vector := randVector(10)
				hnsw.AddPoint(vector, i)
			}

			// Search
			queryVector := randVector(10)
			labels, distances := hnsw.SearchKNN(queryVector, 5)

			if len(labels) != 5 {
				t.Errorf("Expected 5 labels, got %d", len(labels))
			}
			if len(distances) != 5 {
				t.Errorf("Expected 5 distances, got %d", len(distances))
			}

			// Check distances are sorted
			for i := 1; i < len(distances); i++ {
				if distances[i] < distances[i-1] {
					t.Error("Distances are not sorted")
				}
			}
		})
	}
}

// TestHNSW_AddBatchPoints tests adding multiple points in batch
func TestHNSW_AddBatchPoints(t *testing.T) {
	spaceTypes := []string{SpaceL2, SpaceIP, SpaceCosine}

	for _, spaceType := range spaceTypes {
		t.Run(spaceType, func(t *testing.T) {
			hnsw := New(10, 16, 200, 42, 1000, spaceType)
			t.Cleanup(func() { hnsw.Free() })

			vectors := make([][]float32, 100)
			labels := make([]uint32, 100)
			for i := 0; i < 100; i++ {
				vectors[i] = randVector(10)
				labels[i] = uint32(i)
			}

			if !hnsw.AddBatchPoints(vectors, labels, 4) {
				t.Error("AddBatchPoints failed")
			}

			if hnsw.GetCurrentElementCount() != 100 {
				t.Errorf("Expected 100 elements, got %d", hnsw.GetCurrentElementCount())
			}
		})
	}
}

// TestHNSW_SearchBatchKNN tests searching for multiple query vectors
func TestHNSW_SearchBatchKNN(t *testing.T) {
	spaceTypes := []string{SpaceL2, SpaceIP, SpaceCosine}

	for _, spaceType := range spaceTypes {
		t.Run(spaceType, func(t *testing.T) {
			hnsw := New(10, 16, 200, 42, 1000, spaceType)
			t.Cleanup(func() { hnsw.Free() })

			// Add some points
			vectors := make([][]float32, 100)
			labels := make([]uint32, 100)
			for i := 0; i < 100; i++ {
				vectors[i] = randVector(10)
				labels[i] = uint32(i)
			}
			hnsw.AddBatchPoints(vectors, labels, 4)

			// Search batch
			queryVectors := make([][]float32, 10)
			for i := 0; i < 10; i++ {
				queryVectors[i] = randVector(10)
			}

			labelList, distList := hnsw.SearchBatchKNN(queryVectors, 5, 2)

			if len(labelList) != 10 {
				t.Errorf("Expected 10 result sets, got %d", len(labelList))
			}
			if len(distList) != 10 {
				t.Errorf("Expected 10 distance sets, got %d", len(distList))
			}

			for i := 0; i < 10; i++ {
				if len(labelList[i]) != 5 {
					t.Errorf("Expected 5 labels for query %d, got %d", i, len(labelList[i]))
				}
				if len(distList[i]) != 5 {
					t.Errorf("Expected 5 distances for query %d, got %d", i, len(distList[i]))
				}
			}
		})
	}
}

// TestHNSW_SaveLoad tests saving and loading the index
func TestHNSW_SaveLoad(t *testing.T) {
	spaceTypes := []string{SpaceL2, SpaceIP, SpaceCosine}

	for _, spaceType := range spaceTypes {
		t.Run(spaceType, func(t *testing.T) {
			tempDir := os.TempDir()
			tempFile := filepath.Join(tempDir, "hnsw_test_"+spaceType+".bin")
			t.Cleanup(func() { _ = os.Remove(tempFile) })

			// Create and populate index
			hnsw := New(10, 16, 200, 42, 1000, spaceType)
			vectors := make([][]float32, 50)
			labels := make([]uint32, 50)
			for i := 0; i < 50; i++ {
				vectors[i] = randVector(10)
				labels[i] = uint32(i)
				hnsw.AddPoint(vectors[i], labels[i])
			}

			// Save
			if !hnsw.Save(tempFile) {
				t.Error("Save failed")
			}
			hnsw.Free()

			// Load
			loadedHNSW := Load(tempFile, 10, spaceType)
			if loadedHNSW == nil {
				t.Fatal("Load returned nil")
			}
			t.Cleanup(func() { loadedHNSW.Free() })

			// Verify loaded data
			if loadedHNSW.GetCurrentElementCount() != 50 {
				t.Errorf("Expected 50 elements after load, got %d", loadedHNSW.GetCurrentElementCount())
			}

			// Search and verify results match
			queryVector := randVector(10)
			searchLabels, _ := loadedHNSW.SearchKNN(queryVector, 5)
			if len(searchLabels) != 5 {
				t.Errorf("Expected 5 labels after load, got %d", len(searchLabels))
			}
		})
	}
}

// TestHNSW_MarkDeleteUnmarkDelete tests marking and unmarking deletion
func TestHNSW_MarkDeleteUnmarkDelete(t *testing.T) {
	spaceTypes := []string{SpaceL2, SpaceIP, SpaceCosine}

	for _, spaceType := range spaceTypes {
		t.Run(spaceType, func(t *testing.T) {
			hnsw := New(10, 16, 200, 42, 1000, spaceType)
			t.Cleanup(func() { hnsw.Free() })

			vector := randVector(10)
			hnsw.AddPoint(vector, 1)

			// Mark as deleted
			if !hnsw.MarkDelete(1) {
				t.Error("MarkDelete failed")
			}

			if !hnsw.GetLabelIsMarkedDeleted(1) {
				t.Error("Label should be marked as deleted")
			}

			if hnsw.GetDeleteCount() != 1 {
				t.Errorf("Expected 1 deleted element, got %d", hnsw.GetDeleteCount())
			}

			// Unmark as deleted
			if !hnsw.UnmarkDelete(1) {
				t.Error("UnmarkDelete failed")
			}

			if hnsw.GetLabelIsMarkedDeleted(1) {
				t.Error("Label should not be marked as deleted")
			}

			if hnsw.GetDeleteCount() != 0 {
				t.Errorf("Expected 0 deleted elements, got %d", hnsw.GetDeleteCount())
			}
		})
	}
}

// TestHNSW_ResizeIndex tests resizing the index
func TestHNSW_ResizeIndex(t *testing.T) {
	spaceTypes := []string{SpaceL2, SpaceIP, SpaceCosine}

	for _, spaceType := range spaceTypes {
		t.Run(spaceType, func(t *testing.T) {
			hnsw := New(10, 16, 200, 42, 100, spaceType)
			t.Cleanup(func() { hnsw.Free() })

			if hnsw.GetMaxElements() != 100 {
				t.Errorf("Expected max elements 100, got %d", hnsw.GetMaxElements())
			}

			// Resize to 500
			if !hnsw.ResizeIndex(500) {
				t.Error("ResizeIndex failed")
			}

			if hnsw.GetMaxElements() != 500 {
				t.Errorf("Expected max elements 500 after resize, got %d", hnsw.GetMaxElements())
			}

			// Add more points to verify resize worked
			vectors := make([][]float32, 200)
			labels := make([]uint32, 200)
			for i := 0; i < 200; i++ {
				vectors[i] = randVector(10)
				labels[i] = uint32(i)
			}
			hnsw.AddBatchPoints(vectors, labels, 4)

			if hnsw.GetCurrentElementCount() != 200 {
				t.Errorf("Expected 200 elements, got %d", hnsw.GetCurrentElementCount())
			}
		})
	}
}

// TestHNSW_GetVectorByLabel tests retrieving vectors by label
func TestHNSW_GetVectorByLabel(t *testing.T) {
	spaceTypes := []string{SpaceL2, SpaceIP, SpaceCosine}

	for _, spaceType := range spaceTypes {
		t.Run(spaceType, func(t *testing.T) {
			hnsw := New(10, 16, 200, 42, 1000, spaceType)
			t.Cleanup(func() { hnsw.Free() })

			vector := randVector(10)
			hnsw.AddPoint(vector, 1)

			retrievedVector := hnsw.GetVectorByLabel(1)
			if retrievedVector == nil {
				t.Fatal("GetVectorByLabel returned nil")
			}

			if len(retrievedVector) != 10 {
				t.Errorf("Expected vector length 10, got %d", len(retrievedVector))
			}

			// For non-cosine spaces, vectors should match exactly
			if spaceType != SpaceCosine {
				for i := 0; i < 10; i++ {
					if retrievedVector[i] != vector[i] {
						t.Errorf("Vector mismatch at index %d: expected %f, got %f", i, vector[i], retrievedVector[i])
					}
				}
			}

			// Test non-existent label
			nilVector := hnsw.GetVectorByLabel(999)
			if nilVector != nil {
				t.Error("Expected nil for non-existent label")
			}
		})
	}
}

// TestHNSW_UpdatePoint tests updating a point
func TestHNSW_UpdatePoint(t *testing.T) {
	spaceTypes := []string{SpaceL2, SpaceIP, SpaceCosine}

	for _, spaceType := range spaceTypes {
		t.Run(spaceType, func(t *testing.T) {
			hnsw := New(10, 16, 200, 42, 1000, spaceType)
			t.Cleanup(func() { hnsw.Free() })

			vector := randVector(10)
			hnsw.AddPoint(vector, 1)

			newVector := randVector(10)
			if !hnsw.UpdatePoint(newVector, 1, 1.0) {
				t.Error("UpdatePoint failed")
			}

			// Verify the vector was updated
			retrievedVector := hnsw.GetVectorByLabel(1)
			if retrievedVector == nil {
				t.Fatal("GetVectorByLabel returned nil after update")
			}

			// For non-cosine spaces, verify the vector changed
			if spaceType != SpaceCosine {
				changed := false
				for i := 0; i < 10; i++ {
					if retrievedVector[i] != vector[i] {
						changed = true
						break
					}
				}
				if !changed {
					t.Error("Vector was not updated")
				}
			}
		})
	}
}

// TestHNSW_UpdateBatchPoints tests updating multiple points
func TestHNSW_UpdateBatchPoints(t *testing.T) {
	spaceTypes := []string{SpaceL2, SpaceIP, SpaceCosine}

	for _, spaceType := range spaceTypes {
		t.Run(spaceType, func(t *testing.T) {
			hnsw := New(10, 16, 200, 42, 1000, spaceType)
			t.Cleanup(func() { hnsw.Free() })

			// Add initial points
			vectors := make([][]float32, 50)
			labels := make([]uint32, 50)
			for i := 0; i < 50; i++ {
				vectors[i] = randVector(10)
				labels[i] = uint32(i)
				hnsw.AddPoint(vectors[i], labels[i])
			}

			// Update points
			newVectors := make([][]float32, 50)
			updateProbs := make([]float32, 50)
			for i := 0; i < 50; i++ {
				newVectors[i] = randVector(10)
				updateProbs[i] = 1.0
			}

			if !hnsw.UpdateBatchPoints(newVectors, labels, updateProbs, 4) {
				t.Error("UpdateBatchPoints failed")
			}

			// Verify the count remains the same
			if hnsw.GetCurrentElementCount() != 50 {
				t.Errorf("Expected 50 elements after update, got %d", hnsw.GetCurrentElementCount())
			}
		})
	}
}

// TestHNSW_AddPointWithReplace tests adding points with replace deleted support
func TestHNSW_AddPointWithReplace(t *testing.T) {
	spaceTypes := []string{SpaceL2, SpaceIP, SpaceCosine}

	for _, spaceType := range spaceTypes {
		t.Run(spaceType, func(t *testing.T) {
			hnsw := NewWithReplaceDeleted(10, 16, 200, 42, 1000, spaceType)
			t.Cleanup(func() { hnsw.Free() })

			// Add initial points
			for i := uint32(0); i < 10; i++ {
				vector := randVector(10)
				hnsw.AddPoint(vector, i)
			}

			initialCount := hnsw.GetCurrentElementCount()

			// Mark some as deleted
			hnsw.MarkDelete(1)
			hnsw.MarkDelete(3)
			hnsw.MarkDelete(5)

			// Add new points with replace
			for i := uint32(10); i < 13; i++ {
				vector := randVector(10)
				hnsw.AddPointWithReplace(vector, i)
			}

			// With replace deleted, deleted slots are reused so count stays the same
			finalCount := hnsw.GetCurrentElementCount()
			if finalCount != initialCount {
				t.Errorf("Expected count to stay at %d (slots reused), got %d", initialCount, finalCount)
			}

			// Verify the new labels are searchable
			deleteCount := hnsw.GetDeleteCount()
			if deleteCount != 0 {
				t.Errorf("Expected 0 deleted after replace, got %d", deleteCount)
			}
		})
	}
}

// TestHNSW_FreeUnload tests Free and Unload methods
func TestHNSW_FreeUnload(t *testing.T) {
	spaceTypes := []string{SpaceL2, SpaceIP, SpaceCosine}

	for _, spaceType := range spaceTypes {
		t.Run(spaceType, func(t *testing.T) {
			hnsw := New(10, 16, 200, 42, 1000, spaceType)

			vector := randVector(10)
			hnsw.AddPoint(vector, 1)

			// Test Free
			if !hnsw.Free() {
				t.Error("Free failed")
			}

			// After Free, calling Free again should return false
			if hnsw.Free() {
				t.Error("Second Free should return false")
			}

			// After Free, AddPoint should return false
			if hnsw.AddPoint(randVector(10), 99) {
				t.Error("AddPoint should return false after Free")
			}

			// After Free, Save should return false
			if hnsw.Save("/tmp/should_not_exist.bin") {
				t.Error("Save should return false after Free")
			}
		})
	}

	// Test Unload (deprecated alias)
	t.Run("Unload", func(t *testing.T) {
		hnsw := New(10, 16, 200, 42, 1000, SpaceL2)

		vector := randVector(10)
		hnsw.AddPoint(vector, 1)

		// Test Unload
		if !hnsw.Unload() {
			t.Error("Unload failed")
		}
	})
}

// TestHNSW_GetCounts tests GetMaxElements, GetCurrentElementCount, and GetDeleteCount
func TestHNSW_GetCounts(t *testing.T) {
	spaceTypes := []string{SpaceL2, SpaceIP, SpaceCosine}

	for _, spaceType := range spaceTypes {
		t.Run(spaceType, func(t *testing.T) {
			hnsw := New(10, 16, 200, 42, 1000, spaceType)
			t.Cleanup(func() { hnsw.Free() })

			// Initial state
			if hnsw.GetMaxElements() != 1000 {
				t.Errorf("Expected max elements 1000, got %d", hnsw.GetMaxElements())
			}
			if hnsw.GetCurrentElementCount() != 0 {
				t.Errorf("Expected 0 elements initially, got %d", hnsw.GetCurrentElementCount())
			}
			if hnsw.GetDeleteCount() != 0 {
				t.Errorf("Expected 0 deleted elements initially, got %d", hnsw.GetDeleteCount())
			}

			// Add points
			for i := uint32(0); i < 10; i++ {
				vector := randVector(10)
				hnsw.AddPoint(vector, i)
			}

			if hnsw.GetCurrentElementCount() != 10 {
				t.Errorf("Expected 10 elements after adding, got %d", hnsw.GetCurrentElementCount())
			}

			// Mark some as deleted
			hnsw.MarkDelete(1)
			hnsw.MarkDelete(3)

			if hnsw.GetDeleteCount() != 2 {
				t.Errorf("Expected 2 deleted elements, got %d", hnsw.GetDeleteCount())
			}
		})
	}
}

// TestHNSW_SetEf tests setting the ef parameter
func TestHNSW_SetEf(t *testing.T) {
	spaceTypes := []string{SpaceL2, SpaceIP, SpaceCosine}

	for _, spaceType := range spaceTypes {
		t.Run(spaceType, func(t *testing.T) {
			hnsw := New(10, 16, 200, 42, 1000, spaceType)
			t.Cleanup(func() { hnsw.Free() })

			// Add some points
			for i := uint32(0); i < 10; i++ {
				vector := randVector(10)
				hnsw.AddPoint(vector, i)
			}

			// Set ef
			hnsw.SetEf(100)

			// Search should still work
			queryVector := randVector(10)
			labels, _ := hnsw.SearchKNN(queryVector, 5)

			if len(labels) != 5 {
				t.Errorf("Expected 5 labels, got %d", len(labels))
			}
		})
	}
}

// TestHNSW_SetNormalize tests setting the normalize flag
func TestHNSW_SetNormalize(t *testing.T) {
	hnsw := New(10, 16, 200, 42, 1000, SpaceCosine)
	t.Cleanup(func() { hnsw.Free() })

	// By default, cosine space normalizes
	vector := randVector(10)
	hnsw.AddPoint(vector, 1)

	// Set normalize to false (for testing, though this is unusual for cosine)
	hnsw.SetNormalize(false)

	// Add another point
	vector2 := randVector(10)
	hnsw.AddPoint(vector2, 2)

	// Search should still work
	queryVector := randVector(10)
	labels, _ := hnsw.SearchKNN(queryVector, 2)

	if len(labels) != 2 {
		t.Errorf("Expected 2 labels, got %d", len(labels))
	}
}

// TestHNSW_NilIndex tests operations on a freed index (nil internal pointer)
func TestHNSW_NilIndex(t *testing.T) {
	// Create and immediately free to get a struct with nil index
	hnsw := New(10, 16, 200, 42, 100, SpaceL2)
	hnsw.Free()

	// Operations should handle nil index gracefully
	if hnsw.Free() {
		t.Error("Free on freed index should return false")
	}

	if hnsw.Unload() {
		t.Error("Unload on freed index should return false")
	}

	if hnsw.Save("/tmp/test.bin") {
		t.Error("Save on freed index should return false")
	}

	if hnsw.AddPoint([]float32{1, 2, 3}, 1) {
		t.Error("AddPoint on freed index should return false")
	}

	if hnsw.AddPointWithReplace([]float32{1, 2, 3}, 1) {
		t.Error("AddPointWithReplace on freed index should return false")
	}

	labels, dists := hnsw.SearchKNN([]float32{1, 2, 3}, 5)
	if labels != nil || dists != nil {
		t.Error("SearchKNN on freed index should return nil")
	}

	hnsw.SetEf(100)         // Should not panic
	hnsw.SetNormalize(true) // Should not panic

	if hnsw.GetLabelIsMarkedDeleted(1) {
		t.Error("GetLabelIsMarkedDeleted on nil should return false")
	}

	if hnsw.UpdatePoint([]float32{1, 2, 3}, 1, 1.0) {
		t.Error("UpdatePoint on nil should return false")
	}

	vector := hnsw.GetVectorByLabel(1)
	if vector != nil {
		t.Error("GetVectorByLabel on nil should return nil")
	}

	if hnsw.GetCurrentElementCount() != 0 {
		t.Error("GetCurrentElementCount on nil should return 0")
	}
}

// TestHNSW_BatchPointsInvalidInput tests batch operations with invalid inputs
func TestHNSW_BatchPointsInvalidInput(t *testing.T) {
	hnsw := New(10, 16, 200, 42, 1000, SpaceL2)
	t.Cleanup(func() { hnsw.Free() })

	// Mismatched vector and label counts
	vectors := make([][]float32, 10)
	labels := make([]uint32, 5)
	if hnsw.AddBatchPoints(vectors, labels, 2) {
		t.Error("AddBatchPoints with mismatched counts should return false")
	}

	// Invalid coroutines
	vectors = make([][]float32, 10)
	labels = make([]uint32, 10)
	if hnsw.AddBatchPoints(vectors, labels, 0) {
		t.Error("AddBatchPoints with 0 coroutines should return false")
	}

	// UpdateBatchPoints with mismatched counts
	updateProbs := make([]float32, 5)
	if hnsw.UpdateBatchPoints(vectors, labels, updateProbs, 2) {
		t.Error("UpdateBatchPoints with mismatched counts should return false")
	}
}

// identityVector builds a vector whose every coordinate equals id.
// Two distinct ids produce vectors with no shared entries, so any mis-mapping
// between labels and vectors is immediately detectable by SearchKNN /
// GetVectorByLabel.
func identityVector(id, dim int) []float32 {
	v := make([]float32, dim)
	for d := 0; d < dim; d++ {
		v[d] = float32(id + 1) // +1 so that id=0 never produces a zero vector
	}
	return v
}

// batchMappingCases enumerates (size, coroutines) combinations that break the
// symmetries exploited by surviving mutants in the batch-splitting arithmetic
// (e.g. len(vectors) % coroutines == 0, i=0, single-coroutine degenerate path).
var batchMappingCases = []struct {
	name       string
	n          int
	coroutines int
}{
	{"divisible", 100, 4},         // original baseline
	{"indivisible", 103, 4},       // tail block has a remainder
	{"prime_size", 37, 5},         // every shard a different length
	{"single_coroutine", 50, 1},   // degenerate single-shard path
	{"coroutines_gt_items", 3, 4}, // coroutines > len(vectors)
	{"coroutines_eq_items", 8, 8}, // one item per shard
}

// TestHNSW_AddBatchPoints_IdentityMapping verifies that AddBatchPoints
// preserves the 1-to-1 mapping between vectors and labels across several
// (size, coroutines) combinations. It specifically targets mutation-testing
// survivors in the batch-splitting arithmetic: `len/coroutines`, `i*b`,
// `(i+1)*b`, `i == coroutines-1`, `vectors[i*b:end]`, `labels[i*b:end]`.
func TestHNSW_AddBatchPoints_IdentityMapping(t *testing.T) {
	const dim = 8

	for _, c := range batchMappingCases {
		t.Run(c.name, func(t *testing.T) {
			hnsw := New(dim, 16, 200, 42, uint32(c.n+16), SpaceL2)
			t.Cleanup(func() { hnsw.Free() })

			vectors := make([][]float32, c.n)
			labels := make([]uint32, c.n)
			for i := 0; i < c.n; i++ {
				vectors[i] = identityVector(i, dim)
				// Non-sequential labels so that any accidental index/label
				// confusion is also surfaced.
				labels[i] = uint32(i*7 + 1)
			}

			if !hnsw.AddBatchPoints(vectors, labels, c.coroutines) {
				t.Fatalf("AddBatchPoints failed for case %+v", c)
			}

			// Assertion 1: element count matches (kills mutants that drop
			// entire shards by mis-slicing).
			if got := hnsw.GetCurrentElementCount(); got != c.n {
				t.Errorf("element count = %d, want %d", got, c.n)
			}

			// Assertion 2: every label resolves back to the exact vector we
			// inserted. Kills mutants that desynchronise vectors[] and
			// labels[] slices (e.g. vectors[i/b:end] vs labels[i*b:end]).
			for i := 0; i < c.n; i++ {
				got := hnsw.GetVectorByLabel(labels[i])
				if len(got) != dim {
					t.Fatalf("label %d: got len=%d, want %d", labels[i], len(got), dim)
				}
				want := identityVector(i, dim)
				for d := 0; d < dim; d++ {
					if got[d] != want[d] {
						t.Errorf("label=%d dim=%d got=%v want=%v",
							labels[i], d, got[d], want[d])
					}
				}
			}

			// Assertion 3: SearchKNN with the original vector returns the
			// matching label at distance ~0. Kills mutants that drop
			// individual points silently.
			for i := 0; i < c.n; i++ {
				gotLabels, gotDists := hnsw.SearchKNN(vectors[i], 1)
				if len(gotLabels) == 0 {
					t.Errorf("vec[%d]: no result", i)
					continue
				}
				if gotLabels[0] != labels[i] {
					t.Errorf("vec[%d]: got label=%d, want %d",
						i, gotLabels[0], labels[i])
				}
				if gotDists[0] > 1e-4 {
					t.Errorf("vec[%d]: dist=%f, want ~0", i, gotDists[0])
				}
			}
		})
	}
}

// TestHNSW_UpdateBatchPoints_IdentityMapping mirrors the AddBatchPoints test
// for UpdateBatchPoints, targeting the same class of surviving mutants in its
// goroutine splitting logic.
func TestHNSW_UpdateBatchPoints_IdentityMapping(t *testing.T) {
	const dim = 8
	const offset = 1000 // marker to distinguish "updated" from "original" vectors

	for _, c := range batchMappingCases {
		t.Run(c.name, func(t *testing.T) {
			hnsw := New(dim, 16, 200, 42, uint32(c.n+16), SpaceL2)
			t.Cleanup(func() { hnsw.Free() })

			// Seed the index with original vectors.
			labels := make([]uint32, c.n)
			original := make([][]float32, c.n)
			for i := 0; i < c.n; i++ {
				original[i] = identityVector(i, dim)
				labels[i] = uint32(i*7 + 1)
				if !hnsw.AddPoint(original[i], labels[i]) {
					t.Fatalf("AddPoint(%d) failed", labels[i])
				}
			}

			// Build replacement vectors that are clearly distinct from the
			// originals so any mis-mapping becomes visible.
			updated := make([][]float32, c.n)
			probs := make([]float32, c.n)
			for i := 0; i < c.n; i++ {
				updated[i] = identityVector(i+offset, dim)
				probs[i] = 1.0
			}

			if !hnsw.UpdateBatchPoints(updated, labels, probs, c.coroutines) {
				t.Fatalf("UpdateBatchPoints failed for case %+v", c)
			}

			// Count unchanged (kills mutants that skip shards entirely).
			if got := hnsw.GetCurrentElementCount(); got != c.n {
				t.Errorf("element count = %d, want %d", got, c.n)
			}

			// Every label must now resolve to the *updated* vector, not the
			// original. This kills all the `start = i/batchSize`,
			// `end = (i+1)/b`, etc. mutants that leave some labels pointing
			// at their pre-update vectors.
			for i := 0; i < c.n; i++ {
				got := hnsw.GetVectorByLabel(labels[i])
				if len(got) != dim {
					t.Fatalf("label %d: got len=%d, want %d", labels[i], len(got), dim)
				}
				want := updated[i]
				for d := 0; d < dim; d++ {
					if got[d] != want[d] {
						t.Errorf("label=%d dim=%d got=%v want=%v (updated)",
							labels[i], d, got[d], want[d])
					}
				}
			}
		})
	}
}

// TestHNSW_SearchBatchKNN_IdentityMapping verifies that SearchBatchKNN returns
// results in input order and dispatches each query to the correct point.
// Targets surviving mutants in SearchBatchKNN's own goroutine splitting code.
func TestHNSW_SearchBatchKNN_IdentityMapping(t *testing.T) {
	const dim = 8

	for _, c := range batchMappingCases {
		t.Run(c.name, func(t *testing.T) {
			hnsw := New(dim, 16, 200, 42, uint32(c.n+16), SpaceL2)
			t.Cleanup(func() { hnsw.Free() })

			vectors := make([][]float32, c.n)
			labels := make([]uint32, c.n)
			for i := 0; i < c.n; i++ {
				vectors[i] = identityVector(i, dim)
				labels[i] = uint32(i*7 + 1)
				if !hnsw.AddPoint(vectors[i], labels[i]) {
					t.Fatalf("AddPoint(%d) failed", labels[i])
				}
			}

			labelList, distList := hnsw.SearchBatchKNN(vectors, 1, c.coroutines)

			if len(labelList) != c.n {
				t.Fatalf("len(labelList) = %d, want %d", len(labelList), c.n)
			}
			if len(distList) != c.n {
				t.Fatalf("len(distList) = %d, want %d", len(distList), c.n)
			}

			// The i-th query vector must yield the i-th label at distance ~0.
			// Any splitting arithmetic that misroutes queries to goroutines
			// breaks this invariant.
			for i := 0; i < c.n; i++ {
				if len(labelList[i]) == 0 {
					t.Errorf("query %d: no result", i)
					continue
				}
				if labelList[i][0] != labels[i] {
					t.Errorf("query %d: got label=%d, want %d",
						i, labelList[i][0], labels[i])
				}
				if distList[i][0] > 1e-4 {
					t.Errorf("query %d: dist=%f, want ~0", i, distList[i][0])
				}
			}
		})
	}
}

// TestHNSW_MarkDeleteUnmarkDelete_Chain exercises the full MarkDelete →
// IsDeleted → UnmarkDelete → IsDeleted → Search lifecycle for all space types.
// This kills statement/remove mutants that drop the C.markDelete / C.unmarkDelete
// calls and expression/remove mutants on the nil-guard conditions.
func TestHNSW_MarkDeleteUnmarkDelete_Chain(t *testing.T) {
	spaceTypes := []string{SpaceL2, SpaceIP, SpaceCosine}

	for _, spaceType := range spaceTypes {
		t.Run(spaceType, func(t *testing.T) {
			dim := 8
			hnsw := New(dim, 16, 200, 42, 100, spaceType)
			t.Cleanup(func() { hnsw.Free() })

			hnsw.SetEf(50)

			// Add 10 identity vectors: vector[j] = float32(label+1) for all dims.
			for i := 0; i < 10; i++ {
				hnsw.AddPoint(identityVector(i, dim), uint32(i))
			}

			targetLabel := uint32(5)

			// 1. Before delete: should NOT be marked deleted
			if hnsw.GetLabelIsMarkedDeleted(targetLabel) {
				t.Fatal("label should not be marked deleted before MarkDelete")
			}

			// 2. MarkDelete
			if !hnsw.MarkDelete(targetLabel) {
				t.Fatal("MarkDelete returned false")
			}

			// 3. After delete: SHOULD be marked deleted
			if !hnsw.GetLabelIsMarkedDeleted(targetLabel) {
				t.Fatal("label should be marked deleted after MarkDelete")
			}

			// 4. Deleted element count should be 1
			if hnsw.GetDeleteCount() != 1 {
				t.Errorf("GetDeleteCount() = %d, want 1", hnsw.GetDeleteCount())
			}

			// 5. Search should NOT return the deleted label
			queryVec := identityVector(5, dim)
			labels, _ := hnsw.SearchKNN(queryVec, 1)
			if len(labels) > 0 && labels[0] == targetLabel {
				t.Error("SearchKNN returned a deleted label")
			}

			// 6. UnmarkDelete
			if !hnsw.UnmarkDelete(targetLabel) {
				t.Fatal("UnmarkDelete returned false")
			}

			// 7. After undelete: should NOT be marked deleted
			if hnsw.GetLabelIsMarkedDeleted(targetLabel) {
				t.Fatal("label should not be marked deleted after UnmarkDelete")
			}

			// 8. Deleted element count should be 0
			if hnsw.GetDeleteCount() != 0 {
				t.Errorf("GetDeleteCount() = %d, want 0", hnsw.GetDeleteCount())
			}

			// 9. Search SHOULD find the label again (use top-10 to avoid
			//    distance-metric ordering differences across space types).
			labels, _ = hnsw.SearchKNN(queryVec, 10)
			found := false
			for _, l := range labels {
				if l == targetLabel {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("SearchKNN after UnmarkDelete: labels=%v, want %d in results", labels, targetLabel)
			}
		})
	}
}

// TestHNSW_NilIndex_ExtendedMethods ensures that all public methods gracefully
// handle a nil index (after Free). This covers MarkDelete, UnmarkDelete,
// GetLabelIsMarkedDeleted, SetEf, SetNormalize, ResizeIndex, UpdatePoint,
// UpdateBatchPoints, and GetVectorByLabel — some of which were not tested
// in the original nil-safety suite and produced surviving mutants.
func TestHNSW_NilIndex_ExtendedMethods(t *testing.T) {
	hnsw := New(8, 16, 200, 42, 100, SpaceL2)
	hnsw.Free() // index is now nil

	// MarkDelete / UnmarkDelete / GetLabelIsMarkedDeleted
	if hnsw.MarkDelete(0) {
		t.Error("MarkDelete on nil index should return false")
	}
	if hnsw.UnmarkDelete(0) {
		t.Error("UnmarkDelete on nil index should return false")
	}
	if hnsw.GetLabelIsMarkedDeleted(0) {
		t.Error("GetLabelIsMarkedDeleted on nil index should return false")
	}

	// SetEf (void return, just must not panic)
	hnsw.SetEf(50)

	// SetNormalize (void return, just must not panic)
	hnsw.SetNormalize(true)

	// ResizeIndex
	if hnsw.ResizeIndex(200) {
		t.Error("ResizeIndex on nil index should return false")
	}

	// UpdatePoint
	if hnsw.UpdatePoint([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 0, 1.0) {
		t.Error("UpdatePoint on nil index should return false")
	}

	// UpdateBatchPoints
	vecs := [][]float32{{1, 2, 3, 4, 5, 6, 7, 8}}
	if hnsw.UpdateBatchPoints(vecs, []uint32{0}, []float32{1.0}, 1) {
		t.Error("UpdateBatchPoints on nil index should return false")
	}

	// GetVectorByLabel
	if hnsw.GetVectorByLabel(0) != nil {
		t.Error("GetVectorByLabel on nil index should return nil")
	}

	// GetMaxElements / GetCurrentElementCount / GetDeleteCount
	if hnsw.GetMaxElements() != 0 {
		t.Errorf("GetMaxElements on nil index = %d, want 0", hnsw.GetMaxElements())
	}
	if hnsw.GetCurrentElementCount() != 0 {
		t.Errorf("GetCurrentElementCount on nil index = %d, want 0", hnsw.GetCurrentElementCount())
	}
	if hnsw.GetDeleteCount() != 0 {
		t.Errorf("GetDeleteCount on nil index = %d, want 0", hnsw.GetDeleteCount())
	}

	// SearchKNN / SearchBatchKNN
	labels, dists := hnsw.SearchKNN([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 1)
	if labels != nil || dists != nil {
		t.Error("SearchKNN on nil index should return nil, nil")
	}

	// Save
	if hnsw.Save("/tmp/nil_test.bin") {
		t.Error("Save on nil index should return false")
	}

	// Free again (should return false, not panic)
	if hnsw.Free() {
		t.Error("Free on already-freed index should return false")
	}
}
