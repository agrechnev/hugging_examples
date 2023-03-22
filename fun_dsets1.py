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
    print('dset[0]=', dset[0])  # dict
    print("dset['label'][:10]=", dset['label'][:10])  # list
    print('dset[:2]=', dset[:2])  # dict of lists, not Dataset object !!!!
    # Selection: create a Dataset object as a slice
    dset_sel = dset.select(range(10))
    print('dset_sel=', dset_sel)

    # Dataset transformatons
    print('\nDataset Transformation 1')
    # Tokenize the entire dataset
    tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    dset_mapped1 = dset.map(lambda x: tokenizer(x['text']), batched=True)
    # Calculate the length of each text
    dset_mapped2 = dset.map(lambda x: {'text_len': len(x['text'])}, batched=False)
    # Note: the original object dset is not modified !
    print('dset=', dset)
    print('dset_mapped1=', dset_mapped1)
    print('dset_mapped2=', dset_mapped2)

    print('\nDataset Transformation 2')
    # set_transform() adds a transform on the fly
    dset.set_transform(lambda x: tokenizer(x['text']))
    print(dset)  # Still ['text', 'label']
    # Replaces the columns with ['input_ids', 'attention_mask'], ['text', 'label'] are rrmeoved !
    print(dset[0:2].keys())
    dset.set_transform(lambda x: x)  # Clear the transform (don't know if there are better ways)
    print(dset[0:2].keys())

    print('\nDataset Transformation 3')
    print(dset_mapped1.column_names)  # ['text', 'label', 'input_ids', 'attention_mask']
    # Now, let's set Pytorch output format
    dset_mapped1.set_format(type='torch', columns=['label', 'input_ids', 'attention_mask'])
    print(dset_mapped1[0])
    # However, for slices we still get lists of tensors
    # print(dset_mapped1[0:2])
    # Delete a column
    dset_mapped1.remove_columns('text')
    print(dset_mapped1.column_names)  #   ['label', 'input_ids', 'attention_mask']
    print(dset.column_names)   # It is still in dset



########################################################################################################################
if __name__ == '__main__':
    main()
