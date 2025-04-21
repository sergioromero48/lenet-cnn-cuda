# ─── compilers ─────────────────────────────────────────────────
CC      = gcc          # for .c
CXX     = g++          # for .cpp
NVCC    = nvcc         # for .cu

CFLAGS   = -Wall -O2
CXXFLAGS = -Wall -O2
LDFLAGS  = -lm         # add -fopenmp if you use OpenMP in lenet.c

# ─── directories ──────────────────────────────────────────────
SRC_DIR   = src
BUILD_DIR = build

# ─── source files ─────────────────────────────────────────────
CUDA_SRC  = $(SRC_DIR)/lenet.cu   # <- your .cu file
CUDA_OBJ  = $(BUILD_DIR)/lenet_cuda.o

CPU_SRC   = $(SRC_DIR)/lenet.c
CPU_OBJ   = $(BUILD_DIR)/lenet.o

TEST_SRC  = $(SRC_DIR)/test.cpp
TEST_OBJ  = $(BUILD_DIR)/test.o

# -----------------------------------------------------------------
all: directories $(BUILD_DIR)/test

directories:
	@mkdir -p $(BUILD_DIR) data model

# ---------- compile rules ----------------------------------------
$(CUDA_OBJ): $(CUDA_SRC) $(SRC_DIR)/lenet.h $(SRC_DIR)/lenet_cuda.h
	$(NVCC) -c $< -o $@

$(CPU_OBJ): $(CPU_SRC) $(SRC_DIR)/lenet.h
	$(CC) $(CFLAGS) -c $< -o $@

$(TEST_OBJ): $(TEST_SRC)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ---------- link test binary -------------------------------------
$(BUILD_DIR)/test: $(TEST_OBJ) $(CPU_OBJ) $(CUDA_OBJ)
	$(NVCC) -o $@ $^ $(LDFLAGS)

# ---------- convenience targets ----------------------------------
run: all
	@./$(BUILD_DIR)/test data model

clean:
	rm -rf $(BUILD_DIR)
tidy:
	@rm -f $(BUILD_DIR)/*.o
.PHONY: tidy

.PHONY: all run clean directories
