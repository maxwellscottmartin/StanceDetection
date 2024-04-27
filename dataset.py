from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer, RobertaTokenizer, BertTokenizer
import torch, random
import numpy as np

class StanceDataGTE(Dataset):
    def __init__(self, data_name, name='gte1.5-ffnn',
                 max_sen_len=10, max_tok_len=200, max_top_len=5, add_special_tokens=True):
        self.data_name = data_name
        self.data_file = pd.read_csv(data_name)
        self.name = name
        self.max_sen_len = max_sen_len
        self.max_tok_len = max_tok_len
        self.max_top_len = max_top_len
        self.add_special_tokens = add_special_tokens
        self.topic_rep_dict = None
        self.preprocess_data()

    def preprocess_data(self):
        print('preprocessing data {} ...'.format(self.data_name))

        self.data_file['text_idx'] = [[] for _ in range(len(self.data_file))]
        self.data_file['topic_idx'] = [[] for _ in range(len(self.data_file))]
        self.data_file['text_topic'] = [[] for _ in range(len(self.data_file))]
        self.data_file['ori_text'] = ''
        self.data_file['text_l'] = 0
        self.data_file['topic_l'] = 0
        self.data_file['text_mask'] = [[] for _ in range(len(self.data_file))]

        self.tokenizer = AutoTokenizer.from_pretrained('thenlper/gte-base')
        print("processing GTE")
        for i in self.data_file.index:
            row = self.data_file.iloc[i]
            ori_topic = row['topic_str']
            ori_text = row['post']
            text = self.tokenizer(ori_text, max_length=int(self.max_tok_len), truncation=True, padding='max_length', add_special_tokens=self.add_special_tokens)
            topic = self.tokenizer(ori_topic, max_length=int(self.max_top_len), truncation=True, padding='max_length', add_special_tokens=self.add_special_tokens)
            self.data_file.at[i, 'txt_l'] = len(text.input_ids)
            self.data_file.at[i, 'topic_l'] = len(topic.input_ids)
            self.data_file.at[i, 'text_idx'] = text.input_ids
            self.data_file.at[i, 'ori_text'] = ori_text
            self.data_file.at[i, 'topic_idx'] = topic.input_ids
            self.data_file.at[i, 'text_topic'] = joint_dict_gte(self.tokenizer, text, topic)
        print("...finished pre-processing for GTE")
        return

    def get_index(self, word):
        return self.word2i[word] if word in self.word2i else len(self.word2i)

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx, corpus=None):
        row = self.data_file.iloc[idx]

        l = int(row['label'])

        sample = {'text': row['text_idx'], 'topic': row['topic_idx'], 'label': l,
                  'txt_l': row['text_l'], 'top_l': row['topic_l'],
                  'ori_topic': row['topic_str'],
                  'ori_text': row['ori_text'],
                  'text_mask': row['text_mask'],
                  'seen': row['seen?'],
                  'id': row['new_id'],
                  'text_topic': row['text_topic']}
        return sample
    
def joint_dict_gte(tokenizer, text, topic):
    dict = {}
    dict['input_ids'] = tokenizer.build_inputs_with_special_tokens(text.input_ids, topic.input_ids)
    dict['attention_mask'] = [int(t != 0) for t in dict['input_ids']]
    dict['token_type_ids'] = [0] * (2 + len(text.input_ids)) + [1] * (1 + len(topic.input_ids))
    return dict

def prepare_batch_GTE(sample_batched):
    text_lens = np.array([b['txt_l'] for b in sample_batched])
    topic_batch = torch.tensor([b['topic'] for b in sample_batched])
    labels = [b['label'] for b in sample_batched]
    top_lens = [b['top_l'] for b in sample_batched]

    raw_text_batch = [b['ori_text'] for b in sample_batched]
    raw_top_batch = [b['ori_topic'] for b in sample_batched]

    text_batch = torch.tensor([b['text'] for b in sample_batched])

    args = {'text': text_batch, 'topic': topic_batch, 'labels': labels,
            'txt_l': text_lens, 'top_l': top_lens,
            'ori_text': raw_text_batch, 'ori_topic': raw_top_batch}
    
    text_topic_ids = torch.tensor([b['text_topic']['input_ids'] for b in sample_batched])
    text_topic_attn = torch.tensor([b['text_topic']['attention_mask'] for b in sample_batched])
    text_topic_toktype = torch.tensor([b['text_topic']['token_type_ids'] for b in sample_batched])
    args['text_topic_batch'] = {'input_ids': text_topic_ids, 'attention_mask': text_topic_attn, 'token_type_ids': text_topic_toktype}

    return args
class DataSampler:
    def __init__(self, data, batch_size, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        random.seed(0)

        self.indices = list(range(len(data)))
        if shuffle: random.shuffle(self.indices)
        self.batch_num = 0

    def __len__(self):
        return len(self.data)

    def num_batches(self):
        return len(self.data) / float(self.batch_size)

    def __iter__(self):
        self.indices = list(range(len(self.data)))
        if self.shuffle: random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.indices != []:
            idxs = self.indices[:self.batch_size]
            batch = [self.data.__getitem__(i) for i in idxs]
            self.indices = self.indices[self.batch_size:]
            return batch
        else:
            raise StopIteration

    def get(self):
        self.reset()
        return self.__next__()

    def reset(self):
        self.indices = list(range(len(self.data)))
        if self.shuffle: random.shuffle(self.indices)


def setup_helper(args):
    txt_E= args['avg_txt_E'].to('cuda')
    top_E = args['avg_top_E'].to('cuda')
    return txt_E, top_E