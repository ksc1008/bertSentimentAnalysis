import jsonlines
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer
import collections
import matplotlib

skip_corpus = True

# pyplot 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# 데이터 로드 및 처리
train_path = './data/nikluge-ea-2023-train.jsonl'
dev_path = './data/nikluge-ea-2023-dev.jsonl'
test_path = './data/nikluge-ea-2023-test.jsonl'

train_lstText = []
train_lstTarget = []
train_lstY = []

dev_lstText = []
dev_lstTarget = []
dev_lstY = []

test_lstText = []
test_lstTarget = []
test_lstY = []

with jsonlines.open(train_path) as f:
    for line in f.iter():
        train_lstText.append(line['input']['form'])
        train_lstTarget.append(line['input']['target']['form'])
        train_lstY.append(line['output'])

with jsonlines.open(dev_path) as f:
    for line in f.iter():
        dev_lstText.append(line['input']['form'])
        dev_lstTarget.append(line['input']['target']['form'])
        dev_lstY.append(line['output'])

with jsonlines.open(test_path) as f:
    for line in f.iter():
        test_lstText.append(line['input']['form'])
        test_lstTarget.append(line['input']['target']['form'])

cols = ['form', 'target', 'sentiments']


def getList(textList, targetList, yList):
    return zip(textList, targetList, yList)


dfText_train = pd.DataFrame(getList(train_lstText, train_lstTarget, train_lstY), columns=cols)
dfText_train = dfText_train.drop(columns=['sentiments']).join(pd.json_normalize(dfText_train['sentiments']))
dfText_dev = pd.DataFrame(getList(dev_lstText, dev_lstTarget, dev_lstY), columns=cols)
dfText_dev = dfText_dev.drop(columns=['sentiments']).join(pd.json_normalize(dfText_dev['sentiments']))
dfText_test = pd.DataFrame(zip(test_lstText, test_lstTarget), columns=['form', 'target'])

df = pd.concat([dfText_train, dfText_dev])

# 기초적인 EDA 결과
print('n of columns in...\n train : {0}\n dev : {1}\n test : {2}\n'
      .format(len(dfText_train), len(dfText_dev), len(dfText_test)))

print('sentiment counts: ')
for sent in dict(train_lstY[0]).keys():
    print(' {0:13}: {1:5}({2:2.2%})'
          .format(sent, len(df.loc[df[sent] == 'True']),
                  len(df.loc[df[sent] == 'True']) / len(df)))

ax = df['form'].str.len().hist()
ax.set_xlabel('글자 수')
ax.set_ylabel('빈도')
plt.show()

print('average length : {0}'.format(df['form'].str.len().mean()))

def get_df_of_sent(df, sentiment):
    return df.loc[df[sentiment] == 'True']


word_extractor = WordExtractor()
word_score_table = None
corpus = []
l_tokenizer = None
scores = {}

# 토크나이저 초기 설정
if skip_corpus is False:
    word_extractor.load('extractor')

    word_score_table = word_extractor.extract()
    scores = {word: score.cohesion_forward for word, score in word_score_table.items()}

    l_tokenizer = LTokenizer(scores=scores)

def get_ngram(corpus, n, count, title):
    vec = CountVectorizer(ngram_range=(n,n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    x,y = map(list, zip(*words_freq[:count]))
    plt.figure(figsize=(10, 10))
    sb.barplot(x=y, y=x).set(title='{0}ngram corpuses of sentiment{1}'.format(n,title))
    plt.show()

# 최빈 형태소 측정 함수
def get_word_count(df, title):
    global corpus
    forms = df['form'].to_list()
    corpus = []
    corpus_flat = []

    for f in forms:
        corpus.append(l_tokenizer.tokenize(f, flatten=False))

    for c in corpus:
        for elem in c:
            corpus_flat.append(elem[0])

    x, y = [], []
    count = collections.Counter(corpus_flat)
    most = count.most_common()
    for word, count in most[:40]:
        x.append(word)
        y.append(count)

    #plt.figure(figsize=(10, 10))
    #sb.barplot(x=y, y=x).set(title='Most common corpuses of sentiment {0}'.format(title))
    #plt.show()

    corpus_flat = []
    for c in corpus:
        for elem in c:
            corpus_flat.append(' '.join(elem[0]))
            if elem[1] != '':
                corpus_flat.append(elem[1])

    get_ngram(corpus_flat,2,20,title)


def get_target_cont(df, title):
    x, y = [], []
    plt.figure(figsize=(10, 10))
    count = collections.Counter(df["target"])
    most = count.most_common()
    for word, count in most[:40]:
        x.append(word)
        y.append(count)
    plt.figure(figsize=(10, 10))
    sb.barplot(x=y, y=x).set(title='Most common target of sentiment {0}'.format(title))
    plt.show()


for sent in dict(train_lstY[0]).keys():
    get_target_cont(get_df_of_sent(df, sent), sent)

if skip_corpus is False:
    get_word_count(df, "all")

    corpus_lens = []
    for c in corpus:
        corpus_lens.append(len(c))

    ax = pd.DataFrame(corpus_lens)[0].hist()
    ax.set_xlabel('말뭉치 수')
    ax.set_ylabel('빈도')
    plt.show()

    for sent in dict(train_lstY[0]).keys():
        get_word_count(get_df_of_sent(df, sent), sent)
