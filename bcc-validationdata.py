import numpy as np

from latticedata_common import (
    BCC_BASIS,
    NMAX,
    generate_lattice_dataset,
    image_create,
    seed_numba_rng,
    true_lattice_with_metadata as _true_lattice_with_metadata,
    write_lattice_dataset,
)

LATTICE_NAME = 'BCC'
CELL_ATOMS = np.float32(2.0)
BASIS_FRACTION = BCC_BASIS
NVALID = 40000
SEED = 7981


def true_lattice_with_metadata(nmax):
    return _true_lattice_with_metadata(nmax, CELL_ATOMS, BASIS_FRACTION)


def generate_dataset(nsample, seed_value):
    return generate_lattice_dataset(nsample, seed_value, LATTICE_NAME, CELL_ATOMS, BASIS_FRACTION)


if __name__ == '__main__':
    write_lattice_dataset(
        NVALID,
        SEED,
        LATTICE_NAME,
        CELL_ATOMS,
        BASIS_FRACTION,
        'xvalid-large-bcc40k.npy',
        'yvalid-large-bcc40k.npy',
        'metadata-valid-large-bcc40k.npz',
    )
