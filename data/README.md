# Random numeric datasets

- `random_10k.txt` - 10,000 pseudo-random integers uniformly sampled in `[0, 2^31 - 1]`, one per line.
- `random_100k.txt` - 100,000 values with the same format.
- `random_1M.txt` - 1,000,000 values with the same format.
- `random_1B.bin` - 1,000,000,000 values stored in raw binary (unsigned 32-bit little-endian). This format keeps the file at 4,000,000,000 bytes instead of the ~11 GB a text file would require.

To iterate through the billion-value file in Python:

```python
import array
from pathlib import Path

path = Path('data/random_1B.bin')
values = array.array('I')  # unsigned 32-bit
with path.open('rb') as fh:
    values.fromfile(fh, 10)  # read 10 values at a time to avoid huge memory usage
    # process `values` chunk here
```

Adapt the chunk size to your available memory; 20-50 million integers per pass works well on most machines.
