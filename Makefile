# hnswlib-to-go Makefile
# Builds the C++ static library and Go bindings for HNSW vector search.

CXX ?= c++
AR  ?= ar
INCLUDES = -I./third_party/hnswlib
CXXFLAGS = -pthread -std=c++11 $(INCLUDES)
OBJS = hnsw_wrapper.o

# Platform detection
ifeq ($(OS),Windows_NT)
    UNAME_S := Windows
    UNAME_M := $(PROCESSOR_ARCHITECTURE)
    # Windows: MinGW environment
    CXX ?= g++
    CXXFLAGS = -std=c++11 $(INCLUDES)
    LDFLAGS_EXTRA = -lstdc++
    RM = del /Q
    LIB_EXT = .a
else
    UNAME_S := $(shell uname -s)
    UNAME_M := $(shell uname -m)
    RM = rm -rf
    LIB_EXT = .a
    ifeq ($(UNAME_S),Darwin)
        CXXFLAGS += -stdlib=libc++
        LDFLAGS_EXTRA = -lc++
    else ifeq ($(UNAME_S),Linux)
        LDFLAGS_EXTRA = -lstdc++
    endif
endif

# Default: optimized build with native arch
.PHONY: all opt build clean test bench lint mutation mutation-quick help

all: build

opt: CXXFLAGS += -O3 -funroll-loops -march=native
opt: build

# Portable build without -march=native (for CI / cross-compile)
portable: CXXFLAGS += -O2
portable: build

coverage: CXXFLAGS += -O0 -fno-inline -fprofile-arcs --coverage
coverage: build

# ---------- C++ compilation ----------

hnsw_wrapper.o: hnsw_wrapper.h hnsw_wrapper.cc third_party/hnswlib/hnswlib/*.h
	$(CXX) $(CXXFLAGS) -c hnsw_wrapper.cc

libhnsw.a: $(OBJS)
	$(AR) rcs libhnsw.a $(OBJS)

# ---------- Go targets ----------

build: libhnsw.a
	env CGO_CXXFLAGS="$(INCLUDES) -std=c++11" go build

test: build
	env CGO_CXXFLAGS="$(INCLUDES) -std=c++11" go test -v -count=1 -timeout 120s ./...

bench: build
	env CGO_CXXFLAGS="$(INCLUDES) -std=c++11" go test -bench=. -benchmem -benchtime=2s -timeout 300s ./...

lint:
	@command -v golangci-lint >/dev/null 2>&1 && golangci-lint run ./... || echo "golangci-lint not installed, skipping"

# ---------- Mutation testing ----------
# Evaluates test-suite quality by mutating hnsw.go and running the test suite
# against each mutant. Surviving mutants indicate coverage gaps.
#
# Install once:
#   go install github.com/avito-tech/go-mutesting/cmd/go-mutesting@latest
#
# Targets:
#   make mutation        - full config-driven run (uses .go-mutesting.yml)
#   make mutation-quick  - quick CLI run against hnsw.go only (no config file)

mutation: build
	@command -v go-mutesting >/dev/null 2>&1 || { \
		echo "go-mutesting not installed."; \
		echo "Install: go install github.com/avito-tech/go-mutesting/cmd/go-mutesting@latest"; \
		exit 1; \
	}
	env CGO_CXXFLAGS="$(INCLUDES) -std=c++11" \
		go-mutesting \
		--config=.go-mutesting.yml \
		--exec-timeout=240 \
		--disable=arithmetic/bitwise \
		--disable=arithmetic/assign_invert \
		--disable=arithmetic/assignment \
		--disable=branch/else \
		--disable=conditional/negated \
		--disable=expression/comparison \
		--disable=loop/break \
		--disable=loop/condition \
		--disable=loop/range_break \
		--disable=numbers/decrementer \
		./hnsw.go

mutation-quick: build
	@command -v go-mutesting >/dev/null 2>&1 || { \
		echo "go-mutesting not installed."; \
		echo "Install: go install github.com/avito-tech/go-mutesting/cmd/go-mutesting@latest"; \
		exit 1; \
	}
	env CGO_CXXFLAGS="$(INCLUDES) -std=c++11" \
		go-mutesting \
		--exec-timeout=180 \
		./hnsw.go

# ---------- Cross-platform builds ----------
# These targets build the static library for specific OS/arch combinations.
# Requires appropriate cross-compilation toolchains to be installed.

.PHONY: build-linux-amd64 build-linux-arm64 build-darwin-amd64 build-darwin-arm64 build-windows-amd64

build-linux-amd64:
	GOOS=linux GOARCH=amd64 CGO_ENABLED=1 \
	CC=x86_64-linux-gnu-gcc CXX=x86_64-linux-gnu-g++ \
	$(MAKE) CXX=x86_64-linux-gnu-g++ AR=x86_64-linux-gnu-ar build

build-linux-arm64:
	GOOS=linux GOARCH=arm64 CGO_ENABLED=1 \
	CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ \
	$(MAKE) CXX=aarch64-linux-gnu-g++ AR=aarch64-linux-gnu-ar build

build-darwin-amd64:
	GOOS=darwin GOARCH=amd64 CGO_ENABLED=1 \
	$(MAKE) CXXFLAGS="-pthread -std=c++11 $(INCLUDES) -target x86_64-apple-macos11" build

build-darwin-arm64:
	GOOS=darwin GOARCH=arm64 CGO_ENABLED=1 \
	$(MAKE) CXXFLAGS="-pthread -std=c++11 $(INCLUDES) -target arm64-apple-macos11" build

build-windows-amd64:
	GOOS=windows GOARCH=amd64 CGO_ENABLED=1 \
	CC=x86_64-w64-mingw32-gcc CXX=x86_64-w64-mingw32-g++ \
	$(MAKE) CXX=x86_64-w64-mingw32-g++ AR=x86_64-w64-mingw32-ar \
	CXXFLAGS="-std=c++11 $(INCLUDES)" build

# ---------- Cleanup ----------

clean:
ifeq ($(OS),Windows_NT)
	-del /Q *.o libhnsw.a *.gcno *.gcda hnsw.exe 2>nul
else
	rm -rf *.o libhnsw.a *.gcno *.gcda hnsw
endif

# ---------- Help ----------

help:
	@echo "hnswlib-to-go build targets:"
	@echo ""
	@echo "  make build              - Build C++ library and Go package (default)"
	@echo "  make opt                - Build with -O3 and -march=native optimizations"
	@echo "  make portable           - Build without -march=native (CI-friendly)"
	@echo "  make test               - Run unit tests"
	@echo "  make bench              - Run benchmarks"
	@echo "  make lint               - Run golangci-lint (if installed)"
	@echo "  make mutation           - Run mutation testing (uses .go-mutesting.yml)"
	@echo "  make mutation-quick     - Run mutation testing against hnsw.go only"
	@echo "  make clean              - Remove build artifacts"
	@echo ""
	@echo "Cross-platform targets (require cross-compilation toolchains):"
	@echo "  make build-linux-amd64   - Build for Linux x86_64"
	@echo "  make build-linux-arm64   - Build for Linux aarch64"
	@echo "  make build-darwin-amd64  - Build for macOS x86_64"
	@echo "  make build-darwin-arm64  - Build for macOS ARM64"
	@echo "  make build-windows-amd64 - Build for Windows x86_64 (MinGW)"
	@echo ""
	@echo "Detected platform: $(UNAME_S) / $(UNAME_M)"
