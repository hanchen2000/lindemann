from randomdata_common import write_dataset

NTRAIN = 400000
SEED = 92381

if __name__ == '__main__':
    write_dataset(
        NTRAIN,
        SEED,
        'xtrain-large-random400k-2.npy',
        'ytrain-large-random400k-2.npy',
        'metadata-train-large-random400k-2.npz',
    )
