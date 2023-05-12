"""
https://github.com/xunge/pytorch_lmdb_imagenet/blob/master/folder2lmdb.py
"""

import os
import os.path as osp
import pickle

import lmdb
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

NUM_WORKERS = 16


def loads_data(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pickle.loads(buf)


def raw_reader(path):
    with open(path, "rb") as f:
        bin_data = f.read()
    return bin_data


def dumps_data(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pickle.dumps(obj)


def folder2lmdb(dpath, name="train", write_frequency=5000):
    directory = osp.expanduser(dpath)
    print("Loading dataset from %s" % directory)
    dataset = ImageFolder(directory, loader=raw_reader)
    data_loader = DataLoader(dataset, num_workers=NUM_WORKERS, collate_fn=lambda x: x)

    lmdb_path = osp.join(dpath, f"../{name}.lmdb")
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(
        lmdb_path,
        subdir=isdir,
        map_size=1099511627776 * 2,
        readonly=False,
        meminit=False,
        map_async=True,
    )

    txn = db.begin(write=True)
    for idx, data in enumerate(data_loader):
        image, label = data[0]

        txn.put("{}".format(idx).encode("ascii"), dumps_data((image, label)))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = ["{}".format(k).encode("ascii") for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b"__keys__", dumps_data(keys))
        txn.put(b"__len__", dumps_data(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()
