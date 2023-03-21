# By IT-JIM, 2023
# Fun with Datasets framework


import sys

import torch
import datasets
import transformers


########################################################################################################################
def print_it(a, name: str = ''):
    # m = a.float().mean() if isinstance(a, torch.Tensor) else a.mean()
    m = a.mean()
    print(name, a.shape, a.dtype, a.min(), m, a.max())


########################################################################################################################
def main():
    # Let's load a dataset
    # We can download a DatasetDict with splits train, validation, test
    # dset_dict = datasets.load_dataset('rotten_tomatoes')
    # print(dset_dict)

    # Or a single split
    dset = datasets.load_dataset('rotten_tomatoes', split='train')
    print('Dataset Summary')
    print('dset=', dset)
    print('dset.features', dset.features)
    print('len(dset)', len(dset), ', dset.num_rows=', dset.num_rows)
    print('dset.description=', dset.description)

    # Indexing the dataset
    print('\nDataset Indexing')
    print('dset[0]=', dset[0])     # dict
    print("dset['label'][:10]=", dset['label'][:10])   # list
    print('dset[:2]=', dset[:2])  # dict of lists, not Dataset object !!!!
    # Selection: create a Dataset object as a slice
    dset_sel = dset.select(range(10))
    print('dset_sel=', dset_sel)

    # Dataset tokenizing
    print('\nDataset Tokenizing')
    tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
    


########################################################################################################################
if __name__ == '__main__':
    main()