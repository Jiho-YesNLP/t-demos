import sys
import json
import re
import html
from pprint import pprint
from collections import Counter

import code

import pandas as pd
from nltk.corpus import stopwords

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim_models  # don't skip this
import matplotlib.pyplot as plt

# Load jsonl file
FILES = ['data/covid_tweets/ecig_covid_december_to_march.jsonl',
         'data/covid_tweets/ecig_covid_march_to_july.jsonl']
FILES = ['data/covid_tweets/evali_covid_december_to_march.jsonl',
         'data/covid_tweets/evali_covid_march_to_july.jsonl']
STOPWORDS = stopwords.words('english')
# Add initial query words, meta-words, and common words
STOPWORDS.extend(['evali', 'vaping', 'covid', 'coronavirus', 'rt', 'many',
                  'much', 'would'])


def clean_text(text):
    # Remove urls
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    # Remove mentions
    text = re.sub(r'@\S+', '', text)
    # Remove hash symbol, but keep the hash words
    text = re.sub(r'#', ' ', text)
    # decod3 HTML entities
    text = html.unescape(text)
    # run Gensim simple_preprocess
    words = simple_preprocess(text)  # str of document to list of words
    # Remove stopwords, query words, and other common words
    words = [t for t in words if t not in STOPWORDS]
    return words


def find_most_representative_docs(ldamodel, corpus, ids):
    # Classify texts
    sent_topics_df = pd.DataFrame()
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row[0], key=lambda x: (x[1]), reverse=True)

        topic_num, prop_topic = row[0]  # dominant topic
        # wp = ldamodel.show_topic(topic_num)
        # topic_keywords = ', '.join([word for word, prop in wp])
        sent_topics_df = sent_topics_df.append(
            pd.Series([int(topic_num), round(prop_topic, 4)]),
            ignore_index=True
        )
    sent_topics_df.columns = ['Dominant_topic', 'Perc_contribution']
    sent_topics_df = pd.concat([sent_topics_df, pd.Series(ids)], axis=1)

    # Grouped and sorted
    grouped = sent_topics_df.groupby('Dominant_topic')
    sorted_msgs = pd.DataFrame()
    for i, grp in grouped:
        sorted_msgs = pd.concat([sorted_msgs,
                                 grp.sort_values(['Perc_contribution'], ascending=[0]).head(1)],
                                axis=0)
    sorted_msgs.reset_index(drop=True, inplace=True)
    sorted_msgs.columns = ['Topic_num', 'Topic_perc_contrib', 'id']
    return sorted_msgs


class tweet:
    def __init__(self):
        self.text = ''
        self.text_orig = ''
        self.cases = None


data = []
for file in FILES:
    d = [json.loads(l) for l in open(file, encoding='utf-8').read().split('\n')
         if len(l) > 0]
    data.extend(d)

messages = {}
most_rted = Counter()
d = Counter()
for t in data:
    if t['lang'] != 'en':
        continue
    rt = t['retweeted_status'] if 'retweeted_status' in t else None
    qt = t['quoted_status'] if 'quoted_status' in t else None

    tw = tweet()
    text_ = []
    # root-level
    text_root = t['extended_tweet']['full_text'] if t['truncated'] else t['text']

    if rt is None and qt is None:
        text_.append(text_root)
        if t['in_reply_to_status_id'] is None:
            tw.case = 0                    # singletons
        else:
            tw.case = 1                    # replies to a singleton
    elif rt is None and qt is not None:
        text_.append(text_root)
        if t['in_reply_to_status_id'] is None:
            tw.case = 2                    # retweets w comm
        else:
            tw.case = 3                    # replies to (retweet wo comm)
            if qt['truncated']:
                text_.append(qt['extended_tweet']['full_text'])
            else:
                text_.append(qt['text'])
    elif rt is not None and qt is None:
        tw.case = 4                        # retweets wo comm
        if rt['truncated']:
            text_.append(rt['extended_tweet']['full_text'])
        else:
            text_.append(rt['text'])
        most_rted.update([rt['id_str']])
        d.update([t['user']['screen_name']])
    else:
        # replies to (replies or (rt w comm))
        tw.case = 5
        if qt['truncated']:
            text_.append(qt['extended_tweet']['full_text'])
        else:
            text_.append(qt['text'])
        if rt['truncated']:
            text_.append(rt['extended_tweet']['full_text'])
        else:
            text_.append(rt['text'])
    tw.text_orig = ' '.join(text_)
    tw.text = clean_text(' '.join(text_))

    # print(tw.case, t['id_str'])
    messages[t['id_str']] = tw

# Print Stats
print('total', len(messages))
print('singletons', sum([1 for _, t in messages.items() if t.case == 0]))
print('replies', sum([1 for _, t in messages.items() if t.case == 1]))
print('replies to retweets', sum(
    [1 for _, t in messages.items() if t.case == 2]))
print('retweets with comments', sum(
    [1 for _, t in messages.items() if t.case == 3]))
print('rt wo comments', sum([1 for _, t in messages.items() if t.case == 4]))
print('rt of (replies or rt w comments)', sum(
    [1 for _, t in messages.items() if t.case == 5]))
print('dist_rted_tweets', len(most_rted))
print(most_rted.most_common(n=5))
print(d.most_common(n=5))


# # Print messages
# for k in messages:
#     print(k)
#     print('     orig:', messages[k].text_orig)
#     print('     clean:', messages[k].text)
#     print('')

# Create dictionary and corpus
id2word = corpora.Dictionary([tw.text for sid, tw in messages.items()])
corpus = [id2word.doc2bow(tw.text) for sid, tw in messages.items()]

# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=4,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=50,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)

pprint(lda_model.print_topics())

# Visualize
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
pyLDAvis.save_html(vis, 'lda.html')
code.interact(local=locals())

# Classify tweets to a topic
repr_msgs = find_most_representative_docs(
    lda_model, corpus=corpus,
    ids=[sid for sid, tw in messages.items()]
)
print(repr_msgs)

sys.exit()

