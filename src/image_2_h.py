import csv

def convert_csv_row_to_header(csv_path, header_path, size=28):
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        row = next(reader)
        if not row[1].isdigit():
            row = next(reader)  # skip header

        pixels = []
        for i, tok in enumerate(row[1:]):
            try:
                val = int(tok)
                if 0 <= val <= 255:
                    pixels.append(val)
            except ValueError:
                continue

        if len(pixels) != size * size:
            raise ValueError(f"Expected {size*size} pixels, got {len(pixels)}")

    with open(header_path, 'w') as h:
        h.write('#ifndef IMAGE_INPUT_H\n#define IMAGE_INPUT_H\n\n')
        h.write('#include <stdint.h>\n\n')
        h.write(f'#define IMAGE_WIDTH {size}\n#define IMAGE_HEIGHT {size}\n\n')
        h.write(f'static const uint8_t image_input[{size}][{size}] = {{\n')
        for y in range(size):
            row_data = pixels[y * size:(y + 1) * size]
            formatted_row = ', '.join(str(v) for v in row_data)
            h.write(f'    {{{formatted_row}}}')
            h.write(',\n' if y != size - 1 else '\n')
        h.write('};\n\n#endif // IMAGE_INPUT_H\n')

if __name__ == "__main__":
    convert_csv_row_to_header("data/mnist_test-1.csv", "src/image_input.h")
