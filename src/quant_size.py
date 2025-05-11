#!/usr/bin/env python3
import os
from functools import reduce
from operator import mul

# define quantized network layer shapes (same as quantize.py)
layers = {
    'weight0_1': (1, 6, 5, 5),
    'weight2_3': (6, 16, 5, 5),
    'weight4_5': (16, 120, 5, 5),
    'weight5_6': (120 * 1 * 1, 10),
    'bias0_1':   (6,),
    'bias2_3':   (16,),
    'bias4_5':   (120,),
    'bias5_6':   (10,),
}

def prod(shape):
    return reduce(mul, shape, 1)


def main():
    total_params = 0
    print("Quantized network parameter sizes:")
    for name, shape in layers.items():
        count = prod(shape)
        print(f"{name:10s}: {count:6d} params, {count:6d} bytes")
        total_params += count
    print(f"Total parameters: {total_params:,} elements, {total_params:,} bytes")

    # report actual file size
    model_file = os.path.join(os.path.dirname(__file__), '..', 'model', 'model_int8.dat')
    try:
        size = os.path.getsize(model_file)
        print(f"File size on disk: {size:,} bytes")
    except OSError:
        print(f"Could not locate model_int8.dat at {model_file}")

if __name__ == '__main__':
    main()
