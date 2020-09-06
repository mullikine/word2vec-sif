import gensim
from gensim.models import Word2Vec
from fse.models import sif
from fse import IndexedList

import spacy
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

import multiprocessing as mp

def clean_utterance(doc):
    """
    This function takes a SpaCy document and cleans it by removing 
    stop words, non-alphnumerical values, and returns the tokens to their lemmas.

    :doc: Spacy document https://spacy.io/api/doc
    :return: list of lists of strings that are cleaned tokens.
    """
    return [token.lemma_ for token in doc if (token.is_alpha and not token.is_stop)]

def pre_process_docs(docs):
    """
    Pre-processes the list of string input documents using SpaCy.
    It runs in parallel if there are a lot of documents.  

    :docs: list of strings.
    :return: list of Spacy documents https://spacy.io/api/doc
    """
    if len(docs) > 1000:
        batch_size = int(len(docs) / (mp.cpu_count() - 1)) + 1
        return nlp.pipe(docs, batch_size = batch_size, n_process = mp.cpu_count() - 1)
    else:
        return [nlp(doc) for doc in docs]

def multi_process_func(func, values):
    """
    This function takes a function func and an iterable object.
    This function will instantiate n-cores - 1 processes to 
    run the function on the list of values and then join the result 
    and return it. 

    :func: function you want to call on values.
    :values: list of values to pass to func.

    :return: merged list of results from passing values to func
    """
    processes = mp.cpu_count() - 1
    p = mp.Pool(processes=processes)
    if type(values[0]) in [list, tuple]:
        result = p.starmap(func, values)
    else:
        result = p.map(func, values)
    p.close()
    p.join()
    return result

def clean_data(docs):
    """
    Clean the list of string input documents using SpaCy.
    It runs in parallel if there are a lot of documents.  

    :docs: list of strings.
    :return: list of lists of string tokens that have been cleaned.
    """
    nlp_docs = pre_process_docs(docs)
    if len(docs) > 1000: 
        return multi_process_func(clean_utterance, nlp_docs)
    else:
        return [clean_utterance(doc) for doc in nlp_docs]

def create_w2v_model(tokenised_docs, size = 300):
    """
    Trains a word2vec model using Gensim's implementation. 

    :tokenised_docs: list of lists of string tokens.
    :size: vector size to represent the words.
    :return: gensim word2vec model
    """
    # Train word vectors 
    model = Word2Vec(tokenised_docs, size = size, min_count=1, workers= mp.cpu_count() - 1)
    return model

def create_sif_model(model, tokenised_docs):
    """
    Takes a Gensim word2vec model and creates a sif model. 
    implementation can be found here:

    :model: Gensim word2vec model
    :tokenised_docs: list of lists of string tokens.
    :return: word2vec-sif model
    """
    return sif.SIF(model)

def get_doc_vectors(sif_model, indexed_list):
    """
    Trains the word2vec-sif model and returns the document vectors.

    :sif_model: word2vec-sif model
    :indexed_list: a list representation of the documents.
    :return: document embeddings.
    """
    return sif_model.train(indexed_list)

def query_documents(query_string, model, indexed_list, n_results = 10):
    """
    This function querys your list of documents based on a semantic similarity to the
    input string. You will get the n top matching results.

    :query_string: the query string you are searching for
    :model: word2vec-sif model
    :indexed_list: a list representation of the documents.
    :n_results: number of results you want to return
    :return: list of n best matching documents. 
    """
    return model.sv.similar_by_sentence(query_string.split(), \
        model=model, indexable=indexed_list.items, topn=n_results)

def get_sif_model(docs):
    """
    Takes a list of documents and produces a word2vec-sif model with 
    the document embeddings, and the indexed list representation 
    of the documents.

    :docs: list of strings.
    :return: word2vec-sif model with 
    the document embeddings, and the indexed list representation
    """
    cleaned_data = clean_data(docs)
    w2v = create_w2v_model(cleaned_data)
    w2v_sif = create_sif_model(w2v, cleaned_data)
    indexed_list = IndexedList(cleaned_data)
    embeddings = get_doc_vectors(w2v_sif, indexed_list)
    return (w2v_sif, embeddings, indexed_list)

if __name__ == '__main__':
    docs = ['This is one example to see what will happen', \
        'This is another example to see how well it works.', \
        'I have nothing to do with the others. What is going on with this text thingy?']
    model,embeddings,indexed_list = get_sif_model(docs)
    print(query_documents('Example',model,indexed_list,2))