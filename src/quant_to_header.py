#!/usr/bin/env python3
import os
from functools import reduce
from operator import mul

# define quantized network layer shapes
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
    # locate model file and read binary data
    script_dir = os.path.dirname(__file__)
    model_file = os.path.join(script_dir, '..', 'model', 'model_int8.dat')
    try:
        data = open(model_file, 'rb').read()
    except OSError:
        print(f"Error: cannot open {model_file}")
        return

    total_expected = sum(prod(s) for s in layers.values())
    if len(data) != total_expected:
        print(f"Warning: expected {total_expected} bytes, got {len(data)} bytes from model file.")

    # create header file
    header_path = os.path.join(script_dir, 'model.h')
    with open(header_path, 'w') as h:
        guard = '_MODEL_H_'
        h.write(f"#ifndef {guard}\n#define {guard}\n\n")
        h.write("#include <stdint.h>\n\n")

        offset = 0
        for name, shape in layers.items():
            count = prod(shape)
            segment = data[offset:offset+count]
            offset += count
            # write comment and declaration
            h.write(f"// {name} {shape} ({count} bytes)\n")
            h.write(f"const int8_t {name}[{count}] = {{\n")
            # write data in hex, 12 bytes per line
            for i in range(0, len(segment), 12):
                chunk = segment[i:i+12]
                line = ', '.join(f'0x{b:02x}' for b in chunk)
                suffix = ',' if i + 12 < len(segment) else ''
                h.write(f"    {line}{suffix}\n")
            h.write("};\n\n")

        h.write(f"#endif // {guard}\n")
    print(f"Header file generated at {header_path}")

if __name__ == '__main__':
    main()
