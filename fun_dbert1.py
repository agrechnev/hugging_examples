# By IT-JIM, 2023
# Sentiment Analysis (inference only, no training) with DistilBERT, with and without pipelines

import sys
import time

import torch
import transformers
import datasets

MODEL_NAME = 'distilbert-base-uncased-finetuned-sst-2-english'


########################################################################################################################
def print_it(a, name: str = ''):
    # m = a.float().mean() if isinstance(a, torch.Tensor) else a.mean()
    m = a.mean()
    print(name, a.shape, a.dtype, a.min(), m, a.max())


########################################################################################################################
def demo_pipe():
    """Pipeline API demo for Sentiment Analysis"""
    print('\nPIPE !')
    pipe = transformers.pipeline(task='sentiment-analysis', model=MODEL_NAME, device='cpu')

    # Single text inference
    print(pipe('I love Guinea Pigs !'))

    # Batch inference
    # Note: By default pipelines always use batch_size=1 when calling the model
    # THey will just call the mdoel twice !
    print(pipe(['I love Guinea Pigs !', 'I hate you']))


########################################################################################################################
def demo_no_pipe():
    """Transformers API (no pipelines) demo for Sentiment Analysis"""
    print('\nNO PIPE !')

    # Load model from the HUB
    model = transformers.DistilBertForSequenceClassification.from_pretrained(MODEL_NAME)
    # You can also write
    # model = transformers.AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    # BUT THESE WILL NOT WORK: AutoModel, DistilBertModel !!! No classification head !!!

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    # Here AutoTokenizer works fine, the precise tokenizer is DistilBertTokenizerFast

    # Let's try out tokenizer in action
    print('\nTokenizer demo 1')
    emb1 = tokenizer('I love Guinea Pigs !')  # Single string
    print('emb1 =', emb1)
    print(tokenizer.decode(emb1['input_ids']))
    emb2 = tokenizer(['I love Guinea Pigs !', 'I hate you'])  # Or batch
    print('emb2 =', emb2)
    print(tokenizer.batch_decode(emb2['input_ids']))

    # But that will not work with our PyTorch model !
    # We need PyTorch tensors !
    print('\nTokenizer demo 2 : PyTorch tensors')
    emb_pt1 = tokenizer('I love Guinea Pigs !', return_tensors='pt', padding=True, truncation=True, max_length=10)
    print('emb_pt1 =', emb_pt1)
    print(tokenizer.batch_decode(emb_pt1['input_ids']))  # batch_decode() even for single example !!!
    emb_pt2 = tokenizer(['I love Guinea Pigs !', 'I hate you'], return_tensors='pt', padding=True, truncation=True,
                        max_length=10)
    print('emb_pt2 =', emb_pt2)
    print(tokenizer.batch_decode(emb_pt2['input_ids']))

    # Now we run teh actual inference !
    print('\nInference demo')
    out = model(input_ids=emb_pt2['input_ids'], attention_mask=emb_pt2['attention_mask'])
    print('out=', out)
    print('result=', out['logits'].argmax(dim=1))  # Classification result


########################################################################################################################
if __name__ == '__main__':
    # demo_pipe()
    demo_no_pipe()
