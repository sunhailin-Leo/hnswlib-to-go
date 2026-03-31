package hnswgo

import (
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

// benchRandVector generates a random vector with the specified dimension.
func benchRandVector(dim int) []float32 {
	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = rand.Float32()
	}
	return vec
}

// BenchmarkAddPoint_L2 benchmarks single point addition with L2 space.
func BenchmarkAddPoint_L2(b *testing.B) {
	dim := 128
	maxElements := uint32(10000)
	hnsw := New(dim, 16, 200, 42, maxElements, SpaceL2)
	b.Cleanup(func() {
		hnsw.Free()
	})

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		vec := benchRandVector(dim)
		label := uint32(i % int(maxElements))
		hnsw.AddPoint(vec, label)
	}
}

// BenchmarkAddPoint_Cosine benchmarks single point addition with Cosine space.
func BenchmarkAddPoint_Cosine(b *testing.B) {
	dim := 128
	maxElements := uint32(10000)
	hnsw := New(dim, 16, 200, 42, maxElements, SpaceCosine)
	b.Cleanup(func() {
		hnsw.Free()
	})

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		vec := benchRandVector(dim)
		label := uint32(i % int(maxElements))
		hnsw.AddPoint(vec, label)
	}
}

// BenchmarkAddBatchPoints benchmarks batch point addition.
func BenchmarkAddBatchPoints(b *testing.B) {
	dim := 128
	maxElements := uint32(10000)
	batchSize := 1000
	coroutines := 4
	hnsw := New(dim, 16, 200, 42, maxElements, SpaceL2)
	b.Cleanup(func() {
		hnsw.Free()
	})

	vectors := make([][]float32, batchSize)
	labels := make([]uint32, batchSize)
	for i := 0; i < batchSize; i++ {
		vectors[i] = benchRandVector(dim)
		labels[i] = uint32(i)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		hnsw.AddBatchPoints(vectors, labels, coroutines)
	}
}

// BenchmarkSearchKNN_L2 benchmarks KNN search with L2 space.
func BenchmarkSearchKNN_L2(b *testing.B) {
	dim := 128
	maxElements := uint32(10000)
	setupCount := 5000
	topK := 10

	hnsw := New(dim, 16, 200, 42, maxElements, SpaceL2)
	b.Cleanup(func() {
		hnsw.Free()
	})

	// Setup: insert points
	for i := 0; i < setupCount; i++ {
		vec := benchRandVector(dim)
		hnsw.AddPoint(vec, uint32(i))
	}

	queryVec := benchRandVector(dim)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		hnsw.SearchKNN(queryVec, topK)
	}
}

// BenchmarkSearchKNN_Cosine benchmarks KNN search with Cosine space.
func BenchmarkSearchKNN_Cosine(b *testing.B) {
	dim := 128
	maxElements := uint32(10000)
	setupCount := 5000
	topK := 10

	hnsw := New(dim, 16, 200, 42, maxElements, SpaceCosine)
	b.Cleanup(func() {
		hnsw.Free()
	})

	// Setup: insert points
	for i := 0; i < setupCount; i++ {
		vec := benchRandVector(dim)
		hnsw.AddPoint(vec, uint32(i))
	}

	queryVec := benchRandVector(dim)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		hnsw.SearchKNN(queryVec, topK)
	}
}

// BenchmarkSearchBatchKNN benchmarks batch KNN search.
func BenchmarkSearchBatchKNN(b *testing.B) {
	dim := 128
	maxElements := uint32(10000)
	setupCount := 5000
	queryCount := 100
	topK := 10
	coroutines := 4

	hnsw := New(dim, 16, 200, 42, maxElements, SpaceL2)
	b.Cleanup(func() {
		hnsw.Free()
	})

	// Setup: insert points
	for i := 0; i < setupCount; i++ {
		vec := benchRandVector(dim)
		hnsw.AddPoint(vec, uint32(i))
	}

	// Setup: prepare query vectors
	queryVectors := make([][]float32, queryCount)
	for i := 0; i < queryCount; i++ {
		queryVectors[i] = benchRandVector(dim)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		hnsw.SearchBatchKNN(queryVectors, topK, coroutines)
	}
}

// BenchmarkSaveLoad benchmarks Save and Load round-trip performance.
func BenchmarkSaveLoad(b *testing.B) {
	dim := 128
	maxElements := uint32(10000)
	setupCount := 5000

	// Create temporary file for benchmarking
	tmpDir := os.TempDir()
	tmpFile := filepath.Join(tmpDir, "hnsw_benchmark_test.bin")

	hnsw := New(dim, 16, 200, 42, maxElements, SpaceL2)

	// Setup: insert points
	for i := 0; i < setupCount; i++ {
		vec := benchRandVector(dim)
		hnsw.AddPoint(vec, uint32(i))
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		// Save
		hnsw.Save(tmpFile)
		// Free the original index
		hnsw.Free()
		// Load
		hnsw = Load(tmpFile, dim, SpaceL2)
	}

	b.Cleanup(func() {
		if hnsw != nil {
			hnsw.Free()
		}
		_ = os.Remove(tmpFile)
	})
}
