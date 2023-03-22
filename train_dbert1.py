# By IT-JIM, 2023
# Train DistilBert on Rotten Tomatoes

import sys

import torch
import datasets
import transformers

MODEL_NAME = 'distilbert-base-uncased'
DSET_NAME = 'rotten_tomatoes'


########################################################################################################################
def print_it(a, name: str = ''):
    # m = a.float().mean() if isinstance(a, torch.Tensor) else a.mean()
    m = a.mean()
    print(name, a.shape, a.dtype, a.min(), m, a.max())


########################################################################################################################
def demo_train():
    """Train DistilBert"""
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
    if True:
        model.save_pretrained('./trained_model/')
        tokenizer.save_pretrained('./trained_model/')


########################################################################################################################
if __name__ == '__main__':
    demo_train()