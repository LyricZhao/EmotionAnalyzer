import re
from torchtext import data
from torchtext.vocab import Vectors

def text_filter(text):
    text = re.sub('(\d+)|([a-z]+)', ' ', text)
    return text.split()

def label_filter(label):
    label = [int(x) for x in re.findall(r'\d+', label)][1:]
    max_count, max_id = -1, 0
    for i in range(8):
        if label[i] > max_count:
            max_count, max_id = label[i], i
    assert max_id < 8
    return [max_id]

def get_dataset(config):
    text_field, label_field = data.Field(lower=True, batch_first=True), data.Field(batch_first=True, use_vocab=False)
    text_field.tokenize, label_field.tokenize = text_filter, label_filter
    train_dataset, test_dataset = data.TabularDataset.splits(
        path=config['dataset'], format='tsv', skip_header=True,
        train=config['train'], validation=config['test'],
        fields=[('index', None), ('label', label_field), ('text', text_field)]
    )
    if config['preload_w2v']:
        vectors = Vectors(name=config['sgns_model'], cache=config['sgns_model'])
        text_field.build_vocab(train_dataset, test_dataset, vectors=vectors)
    else:
        text_field.build_vocab(train_dataset, test_dataset)
    # label_field.build_vocab(train_dataset, test_dataset)
    train_iter, test_iter = data.Iterator.splits(
        (train_dataset, test_dataset),
        batch_sizes=(config['train_batch_size'], config['test_batch_size']),
        sort_key=lambda x: len(x.text), repeat=False, shuffle=True
    )
    config['vocabulary_size'] = len(text_field.vocab)
    if config['preload_w2v']:
        config.vectors = text_field.vocab.vectors
    return train_iter, test_iter