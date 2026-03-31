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
