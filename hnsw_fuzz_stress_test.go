package hnswgo

import (
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

// TestStressFuzz_AddPointAndSearch simulates fuzz testing for the core
// AddPoint → SearchKNN → UpdatePoint path with randomized parameters.
// This compensates for Go's native fuzz engine not supporting CGO instrumentation.
func TestStressFuzz_AddPointAndSearch(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	iterations := 200

	spaceTypes := []string{SpaceL2, SpaceIP, SpaceCosine}

	for iter := 0; iter < iterations; iter++ {
		dim := rng.Intn(128) + 1
		numPoints := rng.Intn(50) + 1
		topK := rng.Intn(numPoints) + 1
		startLabel := uint32(rng.Intn(1000))
		spaceType := spaceTypes[rng.Intn(len(spaceTypes))]

		index := New(dim, 16, 200, rng.Intn(1000), uint32(numPoints+100), spaceType)
		if index == nil {
			t.Fatalf("iter %d: New returned nil for space %s", iter, spaceType)
		}

		for i := 0; i < numPoints; i++ {
			vec := make([]float32, dim)
			for j := range vec {
				vec[j] = rng.Float32()*2 - 1
			}
			index.AddPoint(vec, startLabel+uint32(i))
		}

		queryVec := make([]float32, dim)
		for j := range queryVec {
			queryVec[j] = rng.Float32()
		}
		labels, dists := index.SearchKNN(queryVec, topK)
		if len(labels) == 0 {
			t.Errorf("iter %d: SearchKNN returned empty (space=%s, dim=%d, n=%d, k=%d)",
				iter, spaceType, dim, numPoints, topK)
		}
		if len(labels) != len(dists) {
			t.Errorf("iter %d: labels/dists mismatch: %d vs %d", iter, len(labels), len(dists))
		}

		// Update a random point
		updateVec := make([]float32, dim)
		for j := range updateVec {
			updateVec[j] = rng.Float32()
		}
		index.UpdatePoint(updateVec, startLabel, 1.0)

		index.Free()
	}
}

// TestStressFuzz_Lifecycle simulates fuzz testing for New → Add → Save → Free → Load → Search → Free
// including deliberate use-after-free attempts.
func TestStressFuzz_Lifecycle(t *testing.T) {
	rng := rand.New(rand.NewSource(123))
	iterations := 100

	for iter := 0; iter < iterations; iter++ {
		dim := rng.Intn(64) + 1
		numPoints := rng.Intn(50) + 1
		topK := rng.Intn(numPoints) + 1

		tmpDir := t.TempDir()
		tmpFile := filepath.Join(tmpDir, "stress_lifecycle.bin")

		index := New(dim, 16, 200, rng.Intn(1000), uint32(numPoints+10), SpaceL2)
		if index == nil {
			t.Fatalf("iter %d: New returned nil", iter)
		}

		for i := 0; i < numPoints; i++ {
			vec := make([]float32, dim)
			for j := range vec {
				vec[j] = rng.Float32()
			}
			index.AddPoint(vec, uint32(i))
		}

		if !index.Save(tmpFile) {
			t.Fatalf("iter %d: Save failed", iter)
		}
		index.Free()

		// Use-after-free safety checks (should not panic)
		index.Free()
		index.AddPoint(make([]float32, dim), 0)
		index.SearchKNN(make([]float32, dim), 1)
		index.Save(tmpFile + ".nope")
		index.SetEf(100)
		index.MarkDelete(0)
		index.UnmarkDelete(0)
		index.GetLabelIsMarkedDeleted(0)
		index.GetVectorByLabel(0)
		_ = index.GetMaxElements()
		_ = index.GetCurrentElementCount()
		_ = index.GetDeleteCount()
		_ = index.ResizeIndex(100)

		// Load and verify
		loaded := Load(tmpFile, dim, SpaceL2)
		if loaded == nil {
			t.Fatalf("iter %d: Load returned nil", iter)
		}

		if loaded.GetCurrentElementCount() != numPoints {
			t.Errorf("iter %d: expected %d elements after load, got %d",
				iter, numPoints, loaded.GetCurrentElementCount())
		}

		labels, _ := loaded.SearchKNN(make([]float32, dim), topK)
		if len(labels) == 0 {
			t.Errorf("iter %d: SearchKNN on loaded index returned empty", iter)
		}

		loaded.Free()
		loaded.Free() // double free safety
	}
}

// TestStressFuzz_EdgeValues tests NaN, Inf, zero vectors, and extreme label values
// across many random iterations to detect CGO boundary crashes.
func TestStressFuzz_EdgeValues(t *testing.T) {
	rng := rand.New(rand.NewSource(456))
	iterations := 500
	dim := 16

	spaceTypes := []string{SpaceL2, SpaceIP, SpaceCosine}

	edgeVecGenerators := []func(dim int) []float32{
		// Zero vector
		func(dim int) []float32 { return make([]float32, dim) },
		// NaN vector
		func(dim int) []float32 {
			vec := make([]float32, dim)
			for i := range vec {
				vec[i] = float32(math.NaN())
			}
			return vec
		},
		// +Inf vector
		func(dim int) []float32 {
			vec := make([]float32, dim)
			for i := range vec {
				vec[i] = float32(math.Inf(1))
			}
			return vec
		},
		// -Inf vector
		func(dim int) []float32 {
			vec := make([]float32, dim)
			for i := range vec {
				vec[i] = float32(math.Inf(-1))
			}
			return vec
		},
		// Mixed special values
		func(dim int) []float32 {
			vec := make([]float32, dim)
			specials := []float32{
				float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1)),
				0, math.MaxFloat32, -math.MaxFloat32, math.SmallestNonzeroFloat32,
			}
			for i := range vec {
				vec[i] = specials[i%len(specials)]
			}
			return vec
		},
	}

	for iter := 0; iter < iterations; iter++ {
		spaceType := spaceTypes[rng.Intn(len(spaceTypes))]
		index := New(dim, 16, 200, 42, 200, spaceType)
		if index == nil {
			t.Fatalf("iter %d: New returned nil", iter)
		}

		// Add a normal point first
		normalVec := make([]float32, dim)
		for i := range normalVec {
			normalVec[i] = float32(i) * 0.1
		}
		index.AddPoint(normalVec, 0)

		// Add edge vector
		genIdx := rng.Intn(len(edgeVecGenerators))
		edgeVec := edgeVecGenerators[genIdx](dim)
		label := uint32(rng.Intn(99)) + 1

		// None of these should panic
		index.AddPoint(edgeVec, label)
		index.SearchKNN(edgeVec, 1)
		index.UpdatePoint(edgeVec, 0, 1.0)
		index.GetVectorByLabel(label)
		index.MarkDelete(label)
		index.UnmarkDelete(label)
		index.GetLabelIsMarkedDeleted(label)

		index.Free()
	}
}

// TestStressFuzz_SaveLoadRoundTrip verifies Save→Load data consistency with
// randomized dimensions, point counts, and space types.
func TestStressFuzz_SaveLoadRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(789))
	iterations := 100

	// Only test non-normalizing spaces for exact vector comparison
	spaceTypes := []string{SpaceL2, SpaceIP}

	for iter := 0; iter < iterations; iter++ {
		dim := rng.Intn(64) + 1
		numPoints := rng.Intn(50) + 1
		spaceType := spaceTypes[rng.Intn(len(spaceTypes))]

		tmpDir := t.TempDir()
		tmpFile := filepath.Join(tmpDir, "stress_roundtrip.bin")

		index := New(dim, 16, 200, rng.Intn(1000), uint32(numPoints+10), spaceType)
		if index == nil {
			t.Fatalf("iter %d: New returned nil (space=%s)", iter, spaceType)
		}

		vectors := make([][]float32, numPoints)
		for i := 0; i < numPoints; i++ {
			vec := make([]float32, dim)
			for j := range vec {
				vec[j] = rng.Float32()*2 - 1
			}
			vectors[i] = vec
			index.AddPoint(vec, uint32(i))
		}

		if !index.Save(tmpFile) {
			t.Fatalf("iter %d: Save failed", iter)
		}
		index.Free()

		loaded := Load(tmpFile, dim, spaceType)
		if loaded == nil {
			t.Fatalf("iter %d: Load returned nil (space=%s)", iter, spaceType)
		}

		if loaded.GetCurrentElementCount() != numPoints {
			t.Errorf("iter %d [%s]: expected %d elements, got %d",
				iter, spaceType, numPoints, loaded.GetCurrentElementCount())
		}

		// Verify every vector matches
		for i := 0; i < numPoints; i++ {
			retrieved := loaded.GetVectorByLabel(uint32(i))
			if retrieved == nil {
				t.Errorf("iter %d [%s]: GetVectorByLabel(%d) returned nil", iter, spaceType, i)
				continue
			}
			for j := 0; j < dim; j++ {
				if retrieved[j] != vectors[i][j] {
					t.Errorf("iter %d [%s]: vector mismatch at label %d, index %d: expected %f, got %f",
						iter, spaceType, i, j, vectors[i][j], retrieved[j])
					break
				}
			}
		}

		loaded.Free()
		_ = os.Remove(tmpFile)
	}
}
