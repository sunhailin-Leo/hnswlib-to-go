package hnswgo

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

// FuzzAddPointAndSearch fuzzes the core AddPoint → SearchKNN → UpdatePoint path
// with random vectors, dimensions clamped to [1, 256], and random labels.
func FuzzAddPointAndSearch(f *testing.F) {
	f.Add(10, 5, uint32(0), uint32(3))
	f.Add(1, 1, uint32(0), uint32(1))
	f.Add(128, 50, uint32(42), uint32(10))

	f.Fuzz(func(t *testing.T, dim int, numPoints int, startLabel uint32, topK uint32) {
		if dim < 1 || dim > 256 {
			t.Skip()
		}
		if numPoints < 1 || numPoints > 200 {
			t.Skip()
		}
		if topK < 1 || topK > uint32(numPoints) {
			t.Skip()
		}

		spaceTypes := []string{SpaceL2, SpaceIP, SpaceCosine}
		for _, spaceType := range spaceTypes {
			index := New(dim, 16, 200, 42, uint32(numPoints+100), spaceType)
			if index == nil {
				t.Fatalf("New returned nil for space %s", spaceType)
			}

			for i := 0; i < numPoints; i++ {
				vec := make([]float32, dim)
				for j := range vec {
					vec[j] = float32(i*dim+j) * 0.001
				}
				index.AddPoint(vec, startLabel+uint32(i))
			}

			queryVec := make([]float32, dim)
			for j := range queryVec {
				queryVec[j] = 0.5
			}
			labels, dists := index.SearchKNN(queryVec, int(topK))
			if len(labels) == 0 {
				t.Errorf("SearchKNN returned empty for space %s", spaceType)
			}
			if len(labels) != len(dists) {
				t.Errorf("labels/dists length mismatch: %d vs %d", len(labels), len(dists))
			}

			// Update a point
			if numPoints > 0 {
				updateVec := make([]float32, dim)
				for j := range updateVec {
					updateVec[j] = float32(j) * 0.01
				}
				index.UpdatePoint(updateVec, startLabel, 1.0)
			}

			index.Free()
		}
	})
}

// FuzzLifecycle fuzzes the New → AddPoint → Save → Free → Load → Search → Free lifecycle
// to detect use-after-free, double-free, and file path issues.
func FuzzLifecycle(f *testing.F) {
	f.Add(10, 20, uint32(5))
	f.Add(1, 1, uint32(0))
	f.Add(64, 100, uint32(99))

	f.Fuzz(func(t *testing.T, dim int, numPoints int, topK uint32) {
		if dim < 1 || dim > 128 {
			t.Skip()
		}
		if numPoints < 1 || numPoints > 100 {
			t.Skip()
		}
		if topK < 1 || topK > uint32(numPoints) {
			t.Skip()
		}

		tmpDir := t.TempDir()
		tmpFile := filepath.Join(tmpDir, "fuzz_lifecycle.bin")

		index := New(dim, 16, 200, 42, uint32(numPoints+10), SpaceL2)
		if index == nil {
			t.Fatal("New returned nil")
		}

		for i := 0; i < numPoints; i++ {
			vec := make([]float32, dim)
			for j := range vec {
				vec[j] = float32(i+j) * 0.01
			}
			index.AddPoint(vec, uint32(i))
		}

		if !index.Save(tmpFile) {
			t.Fatal("Save failed")
		}
		index.Free()

		// Operations on freed index should not panic
		index.Free()
		index.AddPoint(make([]float32, dim), 0)
		index.SearchKNN(make([]float32, dim), 1)
		index.Save(tmpFile + ".nope")

		// Load and verify
		loaded := Load(tmpFile, dim, SpaceL2)
		if loaded == nil {
			t.Fatal("Load returned nil")
		}
		defer loaded.Free()

		if loaded.GetCurrentElementCount() != numPoints {
			t.Errorf("Expected %d elements after load, got %d", numPoints, loaded.GetCurrentElementCount())
		}

		queryVec := make([]float32, dim)
		labels, _ := loaded.SearchKNN(queryVec, int(topK))
		if len(labels) == 0 {
			t.Error("SearchKNN on loaded index returned empty")
		}

		// Double free on loaded index should be safe
		loaded.Free()
		loaded.Free()
	})
}

// FuzzEdgeValues fuzzes with NaN, Inf, zero vectors, and extreme label values
// to ensure no panics or crashes at the CGO boundary.
func FuzzEdgeValues(f *testing.F) {
	f.Add(byte(0), uint32(0))
	f.Add(byte(1), uint32(math.MaxUint32))
	f.Add(byte(2), uint32(12345))

	f.Fuzz(func(t *testing.T, vecType byte, label uint32) {
		dim := 16
		index := New(dim, 16, 200, 42, 100, SpaceL2)
		if index == nil {
			t.Fatal("New returned nil")
		}
		defer index.Free()

		// Add a normal point first so the index is not empty
		normalVec := make([]float32, dim)
		for i := range normalVec {
			normalVec[i] = float32(i) * 0.1
		}
		index.AddPoint(normalVec, 0)

		var edgeVec []float32
		switch vecType % 4 {
		case 0:
			// Zero vector
			edgeVec = make([]float32, dim)
		case 1:
			// NaN vector
			edgeVec = make([]float32, dim)
			for i := range edgeVec {
				edgeVec[i] = float32(math.NaN())
			}
		case 2:
			// +Inf vector
			edgeVec = make([]float32, dim)
			for i := range edgeVec {
				edgeVec[i] = float32(math.Inf(1))
			}
		case 3:
			// -Inf vector
			edgeVec = make([]float32, dim)
			for i := range edgeVec {
				edgeVec[i] = float32(math.Inf(-1))
			}
		}

		// These should not panic — correctness of results is not guaranteed
		// for NaN/Inf inputs, but the process must not crash.
		clampedLabel := (label % 99) + 1 // avoid label 0 collision
		index.AddPoint(edgeVec, uint32(clampedLabel))
		index.SearchKNN(edgeVec, 1)
		index.UpdatePoint(edgeVec, 0, 1.0)
		index.GetVectorByLabel(uint32(clampedLabel))
		index.MarkDelete(uint32(clampedLabel))
		index.UnmarkDelete(uint32(clampedLabel))
		index.GetLabelIsMarkedDeleted(uint32(clampedLabel))
	})
}

// FuzzSaveLoadRoundTrip fuzzes the Save → Load path and verifies data consistency:
// every vector stored before save must be retrievable after load with identical values.
func FuzzSaveLoadRoundTrip(f *testing.F) {
	f.Add(8, 10)
	f.Add(1, 1)
	f.Add(64, 50)

	f.Fuzz(func(t *testing.T, dim int, numPoints int) {
		if dim < 1 || dim > 128 {
			t.Skip()
		}
		if numPoints < 1 || numPoints > 100 {
			t.Skip()
		}

		tmpDir := t.TempDir()

		spaceTypes := []string{SpaceL2, SpaceIP}
		for _, spaceType := range spaceTypes {
			tmpFile := filepath.Join(tmpDir, "fuzz_roundtrip_"+spaceType+".bin")

			index := New(dim, 16, 200, 42, uint32(numPoints+10), spaceType)
			if index == nil {
				t.Fatalf("New returned nil for space %s", spaceType)
			}

			// Store vectors for later comparison
			vectors := make([][]float32, numPoints)
			for i := 0; i < numPoints; i++ {
				vec := make([]float32, dim)
				for j := range vec {
					vec[j] = float32(i*dim+j) * 0.01
				}
				vectors[i] = vec
				index.AddPoint(vec, uint32(i))
			}

			if !index.Save(tmpFile) {
				t.Fatalf("Save failed for space %s", spaceType)
			}
			index.Free()

			loaded := Load(tmpFile, dim, spaceType)
			if loaded == nil {
				t.Fatalf("Load returned nil for space %s", spaceType)
			}

			if loaded.GetCurrentElementCount() != numPoints {
				t.Errorf("[%s] Expected %d elements, got %d", spaceType, numPoints, loaded.GetCurrentElementCount())
			}

			// Verify every vector is retrievable and matches
			for i := 0; i < numPoints; i++ {
				retrieved := loaded.GetVectorByLabel(uint32(i))
				if retrieved == nil {
					t.Errorf("[%s] GetVectorByLabel(%d) returned nil", spaceType, i)
					continue
				}
				if len(retrieved) != dim {
					t.Errorf("[%s] Expected dim %d, got %d for label %d", spaceType, dim, len(retrieved), i)
					continue
				}
				for j := 0; j < dim; j++ {
					if retrieved[j] != vectors[i][j] {
						t.Errorf("[%s] Vector mismatch at label %d, index %d: expected %f, got %f",
							spaceType, i, j, vectors[i][j], retrieved[j])
						break
					}
				}
			}

			loaded.Free()
			_ = os.Remove(tmpFile)
		}
	})
}
