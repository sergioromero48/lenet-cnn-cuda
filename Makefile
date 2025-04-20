CC      = gcc
CFLAGS  = -Wall -O2
LDFLAGS = -lm

SRC_DIR   = src
BUILD_DIR = build

all: directories lenet_train

directories:
	mkdir -p $(BUILD_DIR)
	mkdir -p data
	mkdir -p model

lenet_train: $(SRC_DIR)/main.c $(SRC_DIR)/lenet.c
	$(CC) $(CFLAGS) $^ -o $(BUILD_DIR)/$@ $(LDFLAGS)

run: lenet_train
	@./$(BUILD_DIR)/lenet_train data model

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all directories run clean
