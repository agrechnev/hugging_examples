# By IT-JIM, 2023
# Train GPT-2 with Trainer


import sys

import numpy as np
import torch
import torch.utils.data
import transformers
import tqdm

MODEL_NAME = 'gpt2'
TEXT_CORPUS = 'gpt1_paper.txt'
DEVICE = 'cuda'

TOKEN_ENDOFTEXT = 50256  # '<|endoftext|>
BLOCK_LEN = 512
DEVICE = 'cuda'


########################################################################################################################
def print_it(a, name: str = ''):
    m = a.float().mean() if isinstance(a, torch.Tensor) else a.mean()
    # m = a.mean()
    print(name, a.shape, a.dtype, a.min(), m, a.max())


########################################################################################################################
class MyDset(torch.utils.data.Dataset):
    """A custom dataset that serves 1024-token blocks as input_ids == labels"""
    def __init__(self, data: list[list[int]]):
        self.data = []
        for d in data:
            input_ids = torch.tensor(d, dtype=torch.int64)
            attention_mask = torch.ones(len(d), dtype=torch.int64)
            self.data.append({'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': input_ids})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


########################################################################################################################
def break_text_to_pieces(text_path: str, tokenizer: transformers.PreTrainedTokenizer, block_len: int = 512) -> list[str]:
    """Read a file and convert it to tokenized blocks, edding <|endoftext|> to each block"""
    with open(text_path) as f:
        text = f.read()
    chunk_len0 = block_len - 1  # Leave space for a TOKEN_ENDOFTEXT
    tokens = tokenizer.encode(text)
    blocks = []
    pos = 0
    while pos < len(tokens):
        chunk = tokens[pos: pos + chunk_len0]
        chunk.append(TOKEN_ENDOFTEXT)
        blocks.append(chunk)
        pos += chunk_len0

    if len(blocks[-1]) < block_len:
        del blocks[-1]

    return blocks


########################################################################################################################
def train_val_split(data: list[str], ratio: float):
    n = len(data)
    assert n >= 2
    n_val = max(1, int(n * ratio))
    return data[n_val:], data[:n_val]


########################################################################################################################
def prepare_dsets(text_path: str, tokenizer: transformers.PreTrainedTokenizer, block_len: int):
    """Read the text, prepare the datasets """
    data = break_text_to_pieces(text_path, tokenizer, block_len)
    data_train, data_val = train_val_split(data, 0.2)
    return MyDset(data_train), MyDset(data_val)


########################################################################################################################
def main():
    # Load model and tokenizer
    model = transformers.GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    # model = transformers.GPT2LMHeadModel(transformers.GPT2Config())
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

    # Create datasets and loader
    dset_train, dset_val = prepare_dsets(TEXT_CORPUS, tokenizer, BLOCK_LEN)

    # Train
    training_args = transformers.TrainingArguments(
        output_dir="idiot_save/",
        learning_rate=1e-3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=20,
        evaluation_strategy='epoch',
        save_strategy='no',
        # save_strategy='epoch',
    )
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=dset_train,
        eval_dataset=dset_val,
    )
    trainer.train()

    # Save the model if needed
    if False:
        model.save_pretrained('./trained_model/')
        tokenizer.save_pretrained('./trained_model/')

    # Now our model is trained, try the generation
    text = 'Natural language understanding comprises a wide range of diverse tasks'
    batch = tokenizer([text], return_tensors='pt')
    for k, v in batch.items():
        batch[k] = v.to(DEVICE)
    out = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], max_length=20)
    print('GENERATION=', tokenizer.batch_decode(out.cpu()))


########################################################################################################################
if __name__ == '__main__':
    main()