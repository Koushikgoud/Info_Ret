"""
Simple indexer and search engine built on an inverted-index and the BM25 ranking algorithm.
"""
from fileinput import filename
from genericpath import samefile
import os
from collections import defaultdict, Counter
import pickle
import math
import operator
import code

from tqdm import tqdm
from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from datasets import load_dataset


class Indexer:
    db_file = "./ir.idx"
     # You need to store index file on your disk so that you don't need to                     # re-index when running the program again.
    def __init__(self):
        # # TODO. You will need to create appropriate data structures for the following elements
        self.tok2idx = defaultdict(lambda:len(self.tok2idx))                       # map (token to id)
        self.idx2tok = dict()                     # map (id to token)
        self.postings_lists = dict()                # postings for each word
        self.docs = []                            # encoded document list
        self.raw_ds = None                        # raw documents for result presentation
        self.corpus_stats = { 'avgdl': 0 }        # any corpus-level statistics
        self.stopwords = stopwords.words('english')


        if os.path.exists(self.db_file):
            print("reading db_file")
            index = pickle.load(open(self.db_file, 'rb'))
            self.tok2idx = index.get('tok2idx')
            self.idx2tok = index['idx2tok']
            self.docs = index['docs']
            self.raw_ds = index['raw_ds']
            self.postings_lists =index.get('posting')
            self.corpus_stats['avgdl'] = index.get('avgdl')
        else:
            # TODO. Load CNN/DailyMail dataset, preprocess and create postings lists.
            print("loading CNN dataset")
            ds = load_dataset("cnn_dailymail", '3.0.0', split="test")
            self.raw_ds = ds['article']
            self.clean_text(self.raw_ds)
            self.create_postings_lists()
            
    def clean_text(self, lst_text, query=False):
        # TODO. this function will run in two modes: indexing and query mode.
        # TODO. run simple whitespace-based tokenizer (e.g., RegexpTokenizer)
        # TODO. run lemmatizer (e.g., WordNetLemmatizer)
        # TODO. read documents one by one and process
        tokenizer = RegexpTokenizer("\w+")
        lemmatizer = WordNetLemmatizer()
        print('Cleaning texts.........')
        for l in (lst_text if query else tqdm(lst_text)):
            l = l.lower().strip()
            seq = tokenizer.tokenize(l)
            lst_text = [lemmatizer.lemmatize(l) for l in lst_text]                   # map (token to id)
            lemma = []
            for tok, tag in pos_tag(seq):
                pos = tag[0].lower()
                if pos not in ['a', 'r', 'n', 'v']:
                    pos = 'n'
                lemma.append(lemmatizer.lemmatize(tok, pos))
            self.docs.append(lemma)
        print(lemma)
        return lemma
  
    def create_postings_lists(self):
        # TODO. This creates postings lists of your corpus
        # TODO. While indexing compute avgdl and document frequencies of your vocabulary
        # TODO. Save it, so you don't need to do this again in the next runs.
        # Save
        print("Creating posting lists.")
        avgdl=0
        for di, d in enumerate(tqdm(self.docs)):
            avgdl += len(d)
            for word_index in d:
                if word_index in self.postings_lists:
                    if di not in self.postings_lists[word_index][1]:
                        self.postings_lists[word_index][0] += 1
                    self.postings_lists[word_index][1].append(di)
                    
                else:
                    self.postings_lists[word_index]=[1,[di]]
                #print("this is d",d," and this is di ",di)
                #print("this is self postings ",self.postings_lists[word_index])
                    
        
            self.corpus_stats['avgdl'] = avgdl/len(self.docs)
        index = {
            'avgdl':self.corpus_stats['avgdl'],
            'tok2ix': dict(self.tok2idx),
            'idx2tok':self.idx2tok,
            'docs': self.docs,
            'raw_ds': self.raw_ds,
            'posting': self.postings_lists
        }
        print("index ",index['posting'])
        pickle.dump(index, open(self.db_file, 'wb'))


class SearchAgent:
    def __init__(self, indexer):
        # TODO. set necessary parameters
        self.i = indexer
        self.kl = 1.5                # BM25 parameter k1 for tf saturation
        self.b = 0.75                # BM25 parameter b for document length normalization
        self.avgdl = indexer.corpus_stats.get('avgdl')

    def query(self, q_str):
        # TODO. This is take a query string from a user, run the same clean_text process,
        # TODO. Calculate BM25 scores
        # TODO. Sort  the results by the scores in decsending order
        # TODO. Display the result
        results = {}
        q_idx = self.i.clean_text([q_str], query = True)
        if q_idx is not None:
            for term in q_idx:
                df = self.i.postings_lists[term][0]
                w = math.log2(len(self.i.docs) - df + 0.5) / (df + 0.5)
                noduplicates = list(set(self.i.postings_lists[term][1]))
                for docid in noduplicates:
                    term_frequency_lst = []
                    term_frequency_lst = self.i.postings_lists[term][1]
                    term_frequency = term_frequency_lst.count(docid)
                    document_length = len(self.i.docs[docid])
                    print("this is document length ",document_length," and ki and b ",self.kl," ",self.b," and avgdl is ",self.avgdl)
                    s = (self.kl * term_frequency * w) / (term_frequency + self.kl * (1 - self.b + self.b * document_length / self.avgdl))
                    if docid in results:
                        results[docid] += s
                    else:
                        results[docid] = s
        results = sorted(results.items(), key = operator.itemgetter(1))
        results.reverse()
        if len(results) == 0:
            print("No results returned")
            return None
        else:
            self.display_results(results)


    def display_results(self, results):
        for docid, score in results[:5]:  # print top 5 results
            print(f'\nDocID: {docid}')
            print(f'Score: {score}')
            print('Article:')
            print(self.i.raw_ds[docid])



if __name__ == "__main__":
    i = Indexer()           # instantiate an indexer
    q = SearchAgent(i)      # document retriever
    code.interact(local=dict(globals(), **locals()))