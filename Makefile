# ─── compilers ─────────────────────────────────────────────────
CC      = gcc          # for .c
#CXX     = g++          # for .cpp
#NVCC    = nvcc         # for .cu (disabled CUDA)

CFLAGS   = -Wall -O2
CXXFLAGS = -Wall -O2
LDFLAGS  = -lm         # add -fopenmp if you use OpenMP in lenet.c

# ─── directories ──────────────────────────────────────────────
SRC_DIR   = src
BUILD_DIR = build

# ─── source files ─────────────────────────────────────────────
#CUDA_SRC  = $(SRC_DIR)/lenet.cu   # <- your .cu file (disabled CUDA)
#CUDA_OBJ  = $(BUILD_DIR)/lenet_cuda.o

# Full-precision source for main
CPU_SRC       = $(SRC_DIR)/lenet.c
CPU_OBJ       = $(BUILD_DIR)/lenet_fp.o
# Quantized source for INT8 test
QUANT_SRC     = $(SRC_DIR)/lenet_quant.c
QUANT_OBJ     = $(BUILD_DIR)/lenet_q.o

#TEST_SRC  = $(SRC_DIR)/test.cpp   # disabled test.cpp
#TEST_OBJ  = $(BUILD_DIR)/test.o   # disabled test.o
MAIN_SRC  = $(SRC_DIR)/main.c
MAIN_OBJ  = $(BUILD_DIR)/main.o

# Add test_int8 binary build
TEST8_SRC   = $(SRC_DIR)/test_int8.c
TEST8_OBJ   = $(BUILD_DIR)/test_int8.o

# add python and model dirs
PYTHON    = python3
MODEL_DIR = model
INT8_MODEL = $(MODEL_DIR)/model_int8.dat

# -----------------------------------------------------------------
all: directories $(BUILD_DIR)/main

directories:
	@mkdir -p $(BUILD_DIR) data model

# ---------- compile rules ----------------------------------------
#$(CUDA_OBJ): $(CUDA_SRC) $(SRC_DIR)/lenet.h $(SRC_DIR)/lenet_cuda.h
#	$(NVCC) -c $< -o $@

$(CPU_OBJ): $(CPU_SRC) $(SRC_DIR)/lenet.h
	$(CC) $(CFLAGS) -c $< -o $@

$(QUANT_OBJ): $(QUANT_SRC) $(SRC_DIR)/lenet_quant.h
	$(CC) $(CFLAGS) -c $< -o $@

#$(TEST_OBJ): $(TEST_SRC)
#	$(CXX) $(CXXFLAGS) -c $< -o $@
$(MAIN_OBJ): $(MAIN_SRC)
	$(CC) $(CFLAGS) -c $< -o $@

$(TEST8_OBJ): $(TEST8_SRC) $(SRC_DIR)/lenet_quant.h
	$(CC) $(CFLAGS) -c $< -o $@


# ---------- link main binary -------------------------------------
$(BUILD_DIR)/main: $(CPU_OBJ) $(MAIN_OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)

# ---------- link INT8 test binary ---------------------------------
test_int8: $(QUANT_OBJ) $(TEST8_OBJ) | directories
	$(CC) -o $(BUILD_DIR)/test_int8 $(QUANT_OBJ) $(TEST8_OBJ) $(LDFLAGS)

# ---------- convenience targets ----------------------------------
run: all
	@./$(BUILD_DIR)/main data model

run8: test_int8
	@./$(BUILD_DIR)/test_int8 data model

clean:
	rm -rf $(BUILD_DIR)
tidy:
	@rm -f $(BUILD_DIR)/*.o
.PHONY: tidy

.PHONY: all run clean directories quantize
