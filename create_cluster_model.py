import pandas as pd
from top2vec import Top2Vec

num_topics = 35

model_path = f'./models/top2vec_tweets_and_debates_{num_topics}'
obama_path = f'./data/debate2012/obama_{num_topics}.csv'
romney_path = f'./data/debate2012/romney_{num_topics}.csv'
pol_tweet_path = './data/polTweets/political_social_media.csv'

if __name__ == '__main__':

    df = pd.read_csv('data/debate2012/pres.csv')

    obama = df[df['person'] == 'OBAMA']
    romney = df[df['person'] == 'ROMNEY']

    obama_sentences = obama['dialogue'].tolist()
    romney_sentences = romney['dialogue'].tolist()

    stopwords_f = open('./helpers/stopwords.txt', "r", encoding='utf-8')
    try: 
        content = stopwords_f.read()
        stopwords = content.split('\n')
    finally:
        stopwords_f.close()

    df_pol = pd.read_csv(pol_tweet_path, encoding = "ISO-8859-1")
    df_pol_text = df_pol['text'].to_list()

    model_pol = Top2Vec(df_pol_text, embedding_model='universal-sentence-encoder-large')
    model_pol.add_documents(obama_sentences + romney_sentences)
    
    model_pol.hierarchical_topic_reduction(num_topics=min(num_topics, model_pol.get_num_topics()))
    topic_words, word_scores, topic_nums = model_pol.get_topics(num_topics, reduced=True)

    model_pol.save(model_path)

    o_topics, r_dist, r_topic_words, r_topic_word_scores = model_pol.get_documents_topics([i for i in range(len(df_pol_text), len(df_pol_text) + len(obama_sentences))], reduced=True)
    r_topics, r_dist, r_topic_words, r_topic_word_scores = model_pol.get_documents_topics([i for i in range(len(df_pol_text) + len(obama_sentences), len(df_pol_text) + len(obama_sentences) + len(romney_sentences))], reduced=True)

    topic_strs_o = []             # topic_str
    text_o = []                   # post
    for i in range(len(obama)):
        row = obama.iloc[i]
        topic_strs_o += [o_topics[i]]
        text_o += [row['dialogue']]
    df_o = pd.DataFrame({'topic_str': topic_strs_o, 'post': text_o})
    df_o.to_csv(obama_path, index=False)

    topic_strs_r = []             # topic_str
    text_r = []                   # post
    for i in range(len(romney)):
        row = romney.iloc[i]
        topic_strs_r += [r_topics[i]]
        text_r += [row['dialogue']]
    df_r = pd.DataFrame({'topic_str': topic_strs_r, 'post': text_r})
    df_r.to_csv(romney_path, index=False)

    print("Dumping topics...")
    tops = []

    for i in range(0, len(topic_words)):
        lst = [s for s in topic_words[i] if s not in stopwords]
        tops += [" ".join(lst[0:3])]

    print(tops)