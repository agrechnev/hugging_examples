# By IT-JIM, 2023
# Text generation with GPT-2 (with and without pipelines)


import sys

import torch
import transformers

MODEL_NAME = 'gpt2'


########################################################################################################################
def print_it(a, name: str = ''):
    m = a.float().mean() if isinstance(a, torch.Tensor) else a.mean()
    # m = a.mean()
    print(name, a.shape, a.dtype, a.min(), m, a.max())


########################################################################################################################
def demo_pipe():
    """Pipeline API demo for text generation"""
    print('\nPIPE !')
    pipe = transformers.pipeline(task='text-generation', model=MODEL_NAME, device='cpu')

    # Simple text generation
    print(pipe('The elf queen'))


########################################################################################################################
def demo_no_pipe():
    """Transformers API (no pipelines) demo for Sentiment Analysis"""
    print('\nNO PIPE !')

    # Load model from the HUB
    # But we need to tweak the config, otherwise the temperature will be zero (non-random generation)
    config = transformers.GPT2Config.from_pretrained(MODEL_NAME)
    config.do_sample = config.task_specific_params['text-generation']['do_sample']
    config.max_length = config.task_specific_params['text-generation']['max_length']
    # print(config)
    model = transformers.GPT2LMHeadModel.from_pretrained(MODEL_NAME, config=config)
    # We need a generation head GPT2Model will not DO !!!

    # Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tokenize the input
    enc = tokenizer(['The elf queen'], return_tensors='pt')
    print('enc =', enc)
    print(tokenizer.batch_decode(enc['input_ids']))

    # Generate
    out = model.generate(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'], max_length=20)
    print('out=', out)
    print(tokenizer.batch_decode(out))


########################################################################################################################
def demo_custom_generate():
    """Here we demonstrate custom generation (zero-temperature) for simplicity"""
    # Load model, tokenize input
    model = transformers.GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    enc = tokenizer(['The elf queen'], return_tensors='pt')
    input_ids = enc['input_ids']

    # Add one token at a time, in a loop
    for i in range(20):
        # print_it(input_ids, 'input_ids')
        attention_mask = torch.ones(input_ids.shape, dtype=torch.int64)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)['logits']
        new_id = logits[:, -1, :].argmax(dim=1)       # Generate new ID
        input_ids = torch.cat([input_ids, new_id.unsqueeze(0)], dim=1)   # Add new token

    print(tokenizer.batch_decode(input_ids))  # Decode result


########################################################################################################################
if __name__ == '__main__':
    # demo_pipe()
    demo_no_pipe()
    # demo_custom_generate()