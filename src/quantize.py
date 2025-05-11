#!/usr/bin/env python3
# quantize_lenet.py  – same tensor layout you posted, only writes model_int8.dat

import os, numpy as np

# ----- original layout ---------------------------------------------------
INPUT = 1;  LAYER1 = 6;  LAYER2 = 6;  LAYER3 = 16
LAYER4 = 16; LAYER5 = 120; OUTPUT = 10; K = 5
F0 = 32; F1 = F0-K+1; F2 = F1//2; F3 = F2-K+1; F4 = F3//2; F5 = F4-K+1

layers = {
    'weight0_1': (INPUT, LAYER1, K, K),
    'weight2_3': (LAYER2, LAYER3, K, K),
    'weight4_5': (LAYER4, LAYER5, K, K),
    'weight5_6': (LAYER5 * F5 * F5, OUTPUT),
    'bias0_1':   (LAYER1,),
    'bias2_3':   (LAYER3,),
    'bias4_5':   (LAYER5,),
    'bias5_6':   (OUTPUT,),
}
QMAX = 127  # int8 range

# ----- helpers -----------------------------------------------------------
def load_model(path):
    total = sum(np.prod(s) for s in layers.values())
    data  = np.fromfile(path, dtype=np.float64)
    if data.size != total:
        raise ValueError(f"{path}: expected {total} doubles, got {data.size}")
    return data

def quantize_and_save(fp_path, out_path):
    data   = load_model(fp_path)
    offset = 0
    with open(out_path, 'wb') as f:
        for shape in layers.values():
            size   = int(np.prod(shape))
            tensor = data[offset:offset+size].reshape(shape)
            offset += size
            # use per-layer symmetric quantization based on min/max
            min_val = tensor.min()
            max_val = tensor.max()
            # determine scale (avoid division by zero)
            scale = max(abs(min_val), abs(max_val))
            if scale == 0:
                scale = 1
            # inside quantize_and_save(...)
            q = np.clip( np.round(tensor * 127), -127, 127 ).astype(np.int8)

            f.write(q.tobytes())

# ----- main --------------------------------------------------------------
if __name__ == "__main__":
    fp_file  = os.path.join("model", "model.dat")
    int8_out = os.path.join("model", "model_int8.dat")
    quantize_and_save(fp_file, int8_out)
    print("saved", int8_out)
