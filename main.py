"""
Simple indexer and search engine built on an inverted-index and the BM25 ranking algorithm.
"""
from curses import KEY_A1
from fileinput import filename
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
            index = pickle.load(open(self.db_file, 'rb'))
            self.tok2idx = index['tok2idx']
            self.idx2tok = index['idx2tok']
            self.docs = index['docs']
            self.raw_ds = index['raw_ds']
            self.postings_lists = index['postings']
            self.corpus_stats['avgl'] = index['avgl']
            pass
        else:
            # TODO. Load CNN/DailyMail dataset, preprocess and create postings lists.
            ds = load_dataset("cnn_dailymail", '3.0.0', split="test")
            self.raw_ds = ds['article']
            self.clean_text(self.raw_ds)
            self.create_postings_lists()

    def clean_text(self, lst_text, query=False):
        # TODO. this function will run in two modes: indexing and query mode.
        # TODO. run simple whitespace-based tokenizer (e.g., RegexpTokenizer)
        # TODO. run lemmatizer (e.g., WordNetLemmatizer)
        # TODO. read documents one by one and process
        for l in lst_text:
            # punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
            # if l in punc:  
            #      lst_text = lst_text.replace(l, " ")
            enc_doc=[]  
            l = l.lower().strip()
            token = RegexpTokenizer('\s+', gaps = True)
            lst_text = token.tokenize(l)
            wrdnetlemma = WordNetLemmatizer()
            lst_text = [wrdnetlemma.lemmatize(l) for l in lst_text]                   # map (token to id)
            for w in lst_text:
                self.idx2tok[self.tok2idx[w]] = w
                enc_doc.append(self.tok2idx[w])
            self.docs.append(enc_doc)
           
    def create_postings_lists(self):
        # TODO. This creates postings lists of your corpus
        # TODO. While indexing compute avgdl and document frequencies of your vocabulary
        # TODO. Save it, so you don't need to do this again in the next runs.
        # Save
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
            
        self.corpus_stats['avgdl'] = avgdl/len(self.docs)
        index = {
            'avgdl':self.corpus_stats['avgdl'],
            'tok2idx': dict(self.tok2idx),
            'idx2tok':self.idx2tok,
            'docs': self.docs,
            'raw_ds': self.raw_ds,
            'posting': self.postings_lists,
        }
         #Save the cleaned text out
        pickle.dump(index, open(self.db_file, 'wb'))


class SearchAgent:
    k1 = 1.5                # BM25 parameter k1 for tf saturation
    b = 0.75                # BM25 parameter b for document length normalization

    def __init__(self, indexer):
        # TODO. set necessary parameters
        self.i = indexer
        self.k1 = self.k1 
        self.b = self.b
        # self.avgl = indexer.corpus_stats['avgl']


    def query(self, q_str):
        # TODO. This is take a query string from a user, run the same clean_text process,
        # TODO. Calculate BM25 scores
        # TODO. Sort  the results by the scores in decsending order
        # TODO. Display the result

        results = {}
        if len(results) == 0:
            return None
        else:
            self.display_results(results)


    def display_results(self, results):
        # Decode
        # TODO, the following is an example code, you can change however you would like.
        for docid, score in results[:5]:  # print top 5 results
            print(f'\nDocID: {docid}')
            print(f'Score: {score}')
            print('Article:')
            print(self.i.raw_ds[docid])



if __name__ == "__main__":
    i = Indexer()           # instantiate an indexer
    q = SearchAgent(i)      # document retriever
