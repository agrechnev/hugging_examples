# By IT-JIM, 2023
# Train DistilBert on Rotten Tomatoes, with and without Trainer

import sys

import numpy as np
import torch
import torch.utils.data
import datasets
import transformers
import tqdm

MODEL_NAME = 'distilbert-base-uncased'
DSET_NAME = 'rotten_tomatoes'
DEVICE = 'cuda'


########################################################################################################################
def print_it(a, name: str = ''):
    m = a.float().mean() if isinstance(a, torch.Tensor) else a.mean()
    # m = a.mean()
    print(name, a.shape, a.dtype, a.min(), m, a.max())


########################################################################################################################
def demo_train():
    """Train DistilBert with Trainer"""
    # Load dataset
    dset = datasets.load_dataset(DSET_NAME)
    # Load model and tokenizer
    model = transformers.AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    # Tokenize dataset
    dset = dset.map(lambda x: tokenizer(x['text']), batched=True)
    print('dset.column_names=', dset.column_names)

    # Collator (see demo_collate for more info)
    collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

    # Create small train + val datasets
    # But we want labels 0, 1 to be balanced !
    dset_train = dset['train'].select(list(range(50)) + list(range(5000, 5000 + 50)))
    dset_val = dset['validation'].select(list(range(10)) + list(range(600, 600 + 10)))
    # print(dset_train['label'])
    # print(dset_val['label'])

    # Training argumens
    training_args = transformers.TrainingArguments(
        output_dir="idiot_save/",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        evaluation_strategy='epoch',
        save_strategy='no',
        # save_strategy='epoch',
    )

    # Create trainer
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=dset_train,
        eval_dataset=dset_val,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # Train
    train_result = trainer.train()

    # Save the model if needed
    if False:
        model.save_pretrained('./trained_model/')
        tokenizer.save_pretrained('./trained_model/')


########################################################################################################################
def demo_collator():
    """Demonstrate what a collator is"""
    # Short answer: it's collate_fn for torch.util.data.DataLoader
    # It creates a batch from a list of dataset entries

    # Prepare all stuff as before
    dset = datasets.load_dataset(DSET_NAME)
    # model = transformers.AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    dset = dset.map(lambda x: tokenizer(x['text']), batched=True)
    collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)
    dset_train = dset['train'].select(list(range(50)) + list(range(5000, 5000 + 50)))
    dset_val = dset['validation'].select(list(range(10)) + list(range(600, 600 + 10)))

    # Remove the text column, it makes collator crazy !
    dset_train = dset_train.remove_columns('text')
    batch = collator([dset_train[0], dset_train[1]])
    for k, v in batch.items():
        print_it(v, k)


########################################################################################################################
def demo_train_torch():
    """Train with PyTorch training loop, no Trainer"""
    # Prepare all stuff as before
    dset = datasets.load_dataset(DSET_NAME)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    dset = dset.map(lambda x: tokenizer(x['text']), batched=True)
    collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)
    dset_train = dset['train'].select(list(range(50)) + list(range(5000, 5000 + 50)))
    # dset_val = dset['validation'].select(list(range(10)) + list(range(600, 600 + 10)))

    # Create the dataloader
    dset_train = dset_train.remove_columns('text')
    loader = torch.utils.data.DataLoader(dset_train, batch_size=8, collate_fn=collator, shuffle=True)

    # Test the batch
    if False:
        print('BATCH:')
        batch = next(iter(loader))
        print(type(batch))
        for k, v in batch.items():
            print_it(v, k)
        sys.exit()

    # Device, Optimizer
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    # Training loop, we skip validation for simplicity, easy to add if needed
    for i_epoch in range(10):
        model.train()
        losses = []
        n_tot, n_correct = 0, 0
        for batch in tqdm.tqdm(loader):
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            # The model calculates its own loss, if we supply labels argument
            out = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
            # print(out.keys())   # ['loss', 'logits']
            loss = out['loss']
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            # Predictions for acc
            pred = out['logits'].argmax(dim=1)
            n_correct += torch.sum(pred == batch['labels']).item()
            n_tot += len(pred)

        loss_train = np.mean(losses)
        acc_train = n_correct / n_tot
        print(i_epoch, f': TRAIN loss={loss_train}, acc={acc_train}')


########################################################################################################################
if __name__ == '__main__':
    # demo_train()
    # demo_collator()
    demo_train_torch()