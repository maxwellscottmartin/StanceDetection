import torch, time, json, copy
import torch.nn as nn
from transformers import AutoModel, RobertaModel
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np   

def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
class GTELayer(torch.nn.Module):
    def __init__(self):
        super(GTELayer, self).__init__()

        self.model = AutoModel.from_pretrained('thenlper/gte-base')
        self.dim = 768
        self.static_embeds=True
        self.model = self.model.to('cuda')

    def forward(self, **kwargs):

        text_topic = kwargs['text_topic_batch']

        text_topic['input_ids'] = text_topic['input_ids'].to('cuda')
        text_topic['attention_mask'] = text_topic['attention_mask'].to('cuda')
        text_topic['token_type_ids'] = text_topic['token_type_ids'].to('cuda')

        output = self.model(**text_topic)
        last_hidden = output.last_hidden_state

        max_tok_len = kwargs['top_l'][0] + 1
        last_hidden_txt = average_pool(last_hidden[:, 1:-max_tok_len - 1, :], text_topic['attention_mask'][:, 1:-max_tok_len - 1])
        last_hidden_top = average_pool(last_hidden[:, -max_tok_len:-1, :], text_topic['attention_mask'][:, -max_tok_len:-1])

        embed_args = {'avg_txt_E': last_hidden_txt, 'avg_top_E': last_hidden_top}
        return embed_args

class FFNN(torch.nn.Module):
    def __init__(self, **kwargs):
        super(FFNN, self).__init__()
        self.num_labels = kwargs.get('num_labels', 3)
        self.use_topic = kwargs['add_topic']

        if 'input_dim' in kwargs:
            if self.use_topic:
                in_dim = 2 * kwargs['input_dim']
            else:
                in_dim = kwargs['input_dim']
        else:
            in_dim = kwargs['topic_dim'] + kwargs['text_dim']

        self.model = nn.Sequential(nn.Dropout(p=kwargs['in_dropout_prob']),
                                   nn.Linear(in_dim, kwargs['hidden_size']),
                                   kwargs.get('nonlinear_fn',nn.LeakyReLU()),
                                   nn.Linear(kwargs['hidden_size'], 3,
                                             bias=kwargs.get('bias', True)))


    def forward(self, text, topic):
        if self.use_topic:
            combined_input = torch.cat((text, topic), 1)
        else:
            combined_input = text
        y_pred = self.model(combined_input)
        return y_pred


class TorchModelHandler:
    def __init__(self, num_ckps=10, use_score='f_macro', use_last_batch=True,
                checkpoint_path='./checkpoints/',
                 result_path='./data/stance', **params):
        super(TorchModelHandler, self).__init__()

        self.model = params['model']
        self.embed_model = params['embed_model']
        self.dataloader = params['dataloader']
        self.batching_fn = params['batching_fn']
        self.setup_fn = params['setup_fn']
        self.fine_tune=params.get('fine_tune', False)
        self.save_checkpoints=params.get('save_ckp', False)


        self.num_labels = self.model.num_labels
        self.labels = params.get('labels', None)
        self.name = params['name']
        self.use_last_batch = use_last_batch

        self.loss_function = params['loss_function']
        self.optimizer = params['optimizer']

        self.checkpoint_path = checkpoint_path
        self.checkpoint_num = 0
        self.num_ckps = num_ckps
        self.epoch = 0

        self.result_path = result_path

        self.score_dict = dict()
        self.max_score = 0.
        self.max_lst = []  # to keep top 5 scores
        self.score_key = use_score


        self.device = 'cuda' if params.get('device') is None else 'cuda:'+ params['device']
        self.model = self.model.to(self.device)
        self.loss_function = self.loss_function.to(self.device)

    def save_best(self, data=None, scores=None, data_name=None, class_wise=False):
        if scores is None:
            scores = self.eval_and_print(data=data, data_name=data_name,
                                         class_wise=class_wise)
        scores = copy.deepcopy(scores)  # copy the scores, otherwise storing a pointer which won't track properly

        curr_score = scores[self.score_key]
        score_updated = False
        if len(self.max_lst) < 5:
            score_updated = True
            if len(self.max_lst) > 0:
                prev_max = self.max_lst[-1][0][self.score_key]
            else:
                prev_max = curr_score
            self.max_lst.append((scores, self.epoch - 1))
        elif curr_score > self.max_lst[0][0][self.score_key]:
            score_updated = True
            prev_max = self.max_lst[-1][0][self.score_key]
            self.max_lst[0] = (scores, self.epoch - 1)

        if score_updated:
            self.max_lst = sorted(self.max_lst, key=lambda p: p[0][self.score_key])
            f = open('{}{}.top5_{}.txt'.format(self.result_path, self.name, self.score_key), 'w')
            for p in self.max_lst:
                f.write('Epoch: {}\nScore: {}\nAll Scores: {}\n'.format(p[1], p[0][self.score_key],
                                                                      json.dumps(p[0])))
            print(curr_score, prev_max)
            if curr_score > prev_max:
                if self.save_checkpoints:
                    self.save(num='BEST')

    def save(self, num=None):
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss
        }, '{}ckp-{}-{}.tar'.format(self.checkpoint_path, self.name, num))

        if not self.embed_model.static_embeds:
            torch.save({
                'epoch': self.epoch,
                'model_state_dict': self.embed_model.state_dict(),
            }, '{}ckp-{}-{}.embeddings.tar'.format(self.checkpoint_path, self.name,
                                                   num))

    def load(self, filename, use_cpu=False):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def train_step(self):
        print("[{}] epoch {}".format(self.name, self.epoch))
        self.model.train()
        self.loss = 0.
        start_time = time.time()
        for i_batch, sample_batched in enumerate(self.dataloader):
            self.model.zero_grad()

            y_pred, labels = self.get_pred_with_grad(sample_batched)

            label_tensor = torch.tensor(labels)

            label_tensor = label_tensor.to(self.device)

            graph_loss = self.loss_function(y_pred, label_tensor)

            self.loss += graph_loss.item()

            graph_loss.backward()

            self.optimizer.step()

        end_time = time.time()
        print("   took: {:.1f} min".format((end_time - start_time)/60.))
        self.epoch += 1

    def compute_scores(self, score_fn, true_labels, pred_labels, class_wise, name):
        if self.labels is None:
            labels = [i for i in range(self.num_labels)]
        else:
            labels = self.labels
        n = float(len(labels))

        vals = score_fn(true_labels, pred_labels, labels=labels, average=None)
        self.score_dict['{}_macro'.format(name)] = sum(vals) / n
        if class_wise:
            self.score_dict['{}_anti'.format(name)] = vals[0]
            self.score_dict['{}_pro'.format(name)] = vals[1]
            if n > 2:
                self.score_dict['{}_none'.format(name)] = vals[2]

    def eval_model(self, data=None, class_wise=False):
        pred_labels, true_labels, t2pred, marks = self.predict(data)
        self.score(pred_labels, true_labels, class_wise, t2pred, marks)

        return self.score_dict

    def predict(self, data=None):
        all_y_pred = None
        all_labels = None
        all_marks = None

        self.model.eval()
        self.loss = 0.

        if data is None:
            data = self.dataloader

        t2pred = dict()
        for sample_batched in data:

            with torch.no_grad():
                y_pred, labels = self.get_pred_noupdate(sample_batched)

                label_tensor = torch.tensor(labels)
                label_tensor = label_tensor.to(self.device)
                self.loss += self.loss_function(y_pred, label_tensor).item()

                if isinstance(y_pred, dict):
                    y_pred_arr = y_pred['preds'].detach().cpu().numpy()
                else:
                    y_pred_arr = y_pred.detach().cpu().numpy()
                ls = np.array(labels)

                m = [b['seen'] for b in sample_batched]

                for bi, b in enumerate(sample_batched):
                    t = b['ori_topic']
                    t2pred[t] = t2pred.get(t, ([], []))
                    t2pred[t][0].append(y_pred_arr[bi, :])
                    t2pred[t][1].append(ls[bi])

                if all_y_pred is None:
                    all_y_pred = y_pred_arr
                    all_labels = ls
                    all_marks = m

                else:
                    all_y_pred = np.concatenate((all_y_pred, y_pred_arr), 0)
                    all_labels = np.concatenate((all_labels, ls), 0)
                    all_marks = np.concatenate((all_marks, m), 0)

        for t in t2pred:
            t2pred[t] = (np.argmax(t2pred[t][0], axis=1), t2pred[t][1])

        pred_labels = all_y_pred.argmax(axis=1)
        true_labels = all_labels
        return pred_labels, true_labels, t2pred, all_marks

    def eval_and_print(self, data=None, data_name=None, class_wise=False):
        scores = self.eval_model(data=data, class_wise=class_wise)
        print("Evaling on \"{}\" data".format(data_name))
        for s_name, s_val in scores.items():
            print("{}: {}".format(s_name, s_val))
        return scores

    def score(self, pred_labels, true_labels, class_wise, t2pred, marks, topic_wise=False):
        self.compute_scores(f1_score, true_labels, pred_labels, class_wise, 'f')
        self.compute_scores(precision_score, true_labels, pred_labels, class_wise, 'p')
        self.compute_scores(recall_score, true_labels, pred_labels, class_wise, 'r')

        for v in [1, 0]:
            tl_lst = []
            pl_lst = []
            for m, tl, pl in zip(marks, true_labels, pred_labels):
                if m != v: continue
                tl_lst.append(tl)
                pl_lst.append(pl)
            self.compute_scores(f1_score, tl_lst, pl_lst, class_wise, 'f-{}'.format(v))
            self.compute_scores(precision_score, tl_lst, pl_lst, class_wise, 'p-{}'.format(v))
            self.compute_scores(recall_score, tl_lst, pl_lst, class_wise, 'r-{}'.format(v))

        if topic_wise:
            for t in t2pred:
                self.compute_scores(f1_score, t2pred[t][1], t2pred[t][0], class_wise,
                                    '{}-f'.format(t))

    def get_pred_with_grad(self, sample_batched):
        args = self.batching_fn(sample_batched)

        embed_args = self.embed_model(**args)
        args.update(embed_args)

        y_pred = self.model(*self.setup_fn(args))

        labels = args['labels']

        return y_pred, labels

    def get_pred_noupdate(self, sample_batched):
        args = self.batching_fn(sample_batched)

        with torch.no_grad():
            embed_args = self.embed_model(**args)
            args.update(embed_args)

            y_pred = self.model(*self.setup_fn(args))

            labels = args['labels']

        return y_pred, labels