import pandas as pd

import model as md
import dataset as ds
import torch.optim as optim
import torch.nn as nn

n = 35

model_path_stance = './models/gte-base-ffnn-FINAL.tar'
obama_path = f'./data/debate2012/obama_{n}.csv'
obama_mod_path = f'./data/debate2012/obama_{n}_mod.csv'
romney_path = f'./data/debate2012/romney_{n}.csv'
romney_mod_path = f'./data/debate2012/romney_{n}_mod.csv'

topics = ['sen pass agree', 'obamacare healthcare insurance', 'jobs economy job', 'meeting meet county', 'spending bipartisan obamacare', 'veterans honored military', 'meeting discuss interview', 'meeting meet met', 'syria war obama', 'president obama legislation', 'obamacare legislation bipartisan', 'washington honored county', 'energy obama obamacare', 'education students school', 'obama president congressman', 'syria obama war', 'congress bipartisan congressional', 'immigration reform bipartisan', 'congrats congratulations honored', 'prayers obamacare insurance', 'workers obamacare jobs', 'rt obamacare icymi', 'congressional congratulations congrats', 'senator senate congressman', 'veterans congressman military', 'businesses business economic', 'icymi congrats congratulations', 'news read listen', 'congratulations congrats happy', 'obama obamacare administration', 'vote voted republicans', 'obamacare congressional congress', 'obama honored honor', 'irs obamacare tax', 'obamacare healthcare seniors']

# update this to use config file, tired of copy/paste
batch_size = 64
checkpoint_path = 'checkpoints/'
epochs = 9
result_path = 'results/'
hidden_size = 275
in_dropout = 0.3
lr = .0006
max_tok_len = 200
max_top_len = 20
early_stop = 0
name = 'gte-base-ffnn'
bias=False

def gen_dataloaders( obama_path, obama_mod_path, romney_mod_path, romney_path, topics ):
    obama = pd.read_csv(obama_path)
    romney = pd.read_csv(romney_path)

    obama['label'] = [2 for _ in range(len(obama))]
    romney['label'] = [2 for _ in range(len(romney))]

    obama['seen?'] = [0 for _ in range(len(obama))]
    romney['seen?'] = [0 for _ in range(len(romney))]

    obama['new_id'] = [i for i in range(0, len(obama))]
    romney['new_id'] = [i for i in range(len(obama), len(obama) + len(romney))]

    obama['ori_topic'] = ['' for _ in range(len(obama))]
    romney['ori_topic'] = ['' for _ in range(len(romney))]

    for i in range(len(obama)):
        row = obama.iloc[i]
        obama.at[i, 'topic_str'] = topics[int(row['topic_str'])]
        obama.at[i, 'ori_topic'] = topics[int(row['topic_str'])]

    for i in range(len(romney)):
        row = romney.iloc[i]
        romney.at[i, 'topic_str'] = topics[int(row['topic_str'])]
        romney.at[i, 'ori_topic'] = topics[int(row['topic_str'])]

    obama.to_csv(obama_mod_path, index=False)
    romney.to_csv(romney_mod_path, index=False)

    obama_data = ds.StanceDataGTE(obama_mod_path, max_tok_len=max_tok_len,
                                           max_top_len=max_top_len,
                                           add_special_tokens=False)
    obama_dataloader = ds.DataSampler(obama_data, batch_size=batch_size, shuffle=False)

    romney_data = ds.StanceDataGTE(romney_mod_path, max_tok_len=max_tok_len,
                                           max_top_len=max_top_len,
                                           add_special_tokens=False)
    romney_dataloader = ds.DataSampler(romney_data, batch_size=batch_size, shuffle=False)

    return obama_dataloader, romney_dataloader, obama, romney



if __name__ == '__main__':
    obama_dataloader, romney_dataloader, obama_df, romney_df = gen_dataloaders(obama_path=obama_path, obama_mod_path=obama_mod_path, romney_path=romney_path, romney_mod_path=romney_mod_path, topics=topics)

    input_layer = md.GTELayer()

    setup_fn = ds.setup_helper

    loss_fn = nn.CrossEntropyLoss()
    model = md.FFNN(input_dim=input_layer.dim, in_dropout_prob=in_dropout,
                        hidden_size=hidden_size,
                        add_topic=True, bias=bias)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    kwargs = {'model': model, 'embed_model': input_layer, 'dataloader': None,
                  'batching_fn': ds.prepare_batch_GTE, 
                  'name': name,
                  'loss_function': loss_fn,
                  'optimizer': optimizer,
                  'setup_fn': setup_fn,
                  'fine_tune': False}

    model_handler = md.TorchModelHandler(checkpoint_path=checkpoint_path,
                                         result_path=result_path,
                                         **kwargs)
    
    model_handler.load(model_path_stance)

    preds_o, _, _, _ = model_handler.predict(obama_dataloader)
    preds_r, _, _, _ = model_handler.predict(romney_dataloader)

    obama_df['pred_label'] = preds_o
    romney_df['pred_label'] = preds_r

    obama_df.to_csv(obama_mod_path, index=False)
    romney_df.to_csv(romney_mod_path, index=False)