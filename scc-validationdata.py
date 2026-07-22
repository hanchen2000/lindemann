import numpy as np

from latticedata_common import (
    SCC_BASIS,
    NMAX,
    generate_lattice_dataset,
    image_create,
    seed_numba_rng,
    true_lattice_with_metadata as _true_lattice_with_metadata,
    write_lattice_dataset,
)

LATTICE_NAME = 'SCC'
CELL_ATOMS = np.float32(1.0)
BASIS_FRACTION = SCC_BASIS
NVALID = 40000
SEED = 88482


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
        'xvalid-large-scc40k.npy',
        'yvalid-large-scc40k.npy',
        'metadata-valid-large-scc40k.npz',
    )
