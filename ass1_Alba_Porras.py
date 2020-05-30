import nltk, difflib
import numpy as np
import matplotlib.pyplot as plt
from nltk.util import ngrams
from nltk.corpus import treebank_raw, treebank_chunk
nltk.download('treebank')

pattern = r'''(?x)                   # set flag to allow verbose regexps
     (?:[A-Z]\.|[a-z]\.)+            # acronyms, e.g., U.S.A.
   | [A-Z][a-z]{,3}\.                # abbreviations, e.g., Nov.
   | \d+(?:-\w+)+                    # number-word with interval hyphen, e.g., 12-month
   | \d+[a-z]|\d+/\d+                # number + letter sequences (e.g. 20s ) or number sequences with / in between (e.g. 11/02/2020)
   | \d+(?:\.?\,?\d+)+               # number with decimals or high numbers e.g. 2.000.000
   | \w+&\w+                         # currency and percentages, e.g., $12.40, 82%
   | \w+(?=n't)|n't|\w+(?=')|'\w+    # contractions
   | \w+(?:-\w+)*(?:/\w+)*           # words with optional internal hyphens
   | \.\.\.                          # ellipsis 
   | [][\.$,;"'?():-_`{}%&#]         # these are separate tokens; includes ], [
 '''

def get_corpus_text(nr_files=199):
    """Returns the raw corpus as a long string.
    'nr_files' says how much of the corpus is returned;
    default is 199, which is the whole corpus.
    """
    fileids = nltk.corpus.treebank_raw.fileids()[:nr_files]
    corpus_text = nltk.corpus.treebank_raw.raw(fileids)
    corpus_text = corpus_text.replace(".START", "")
    return corpus_text

def fix_treebank_tokens(tokens):
    """Replace tokens so that they are similar to the raw corpus text."""
    return [token.replace("''", '"').replace("``", '"').replace(r"\/", "/") for token in tokens]

def get_gold_tokens(nr_files=199):
    """Returns the gold corpus as a list of strings.
    'nr_files' says how much of the corpus is returned;
    default is 199, which is the whole corpus.
    """
    fileids = nltk.corpus.treebank_chunk.fileids()[:nr_files]
    gold_tokens = nltk.corpus.treebank_chunk.words(fileids)
    return fix_treebank_tokens(gold_tokens)

def tokenize_corpus(text, pattern):
    """tokenize the text input with the regular expression in pattern"""
    tokens = nltk.regexp_tokenize(text, pattern)
    return tokens

def evaluate_tokenization(test_tokens, gold_tokens):
    """Finds the chunks where test_tokens differs from gold_tokens.
    Prints the errors and calculates similarity measures.
    """
    matcher = difflib.SequenceMatcher()
    matcher.set_seqs(test_tokens, gold_tokens)
    error_chunks = true_positives = false_positives = false_negatives = 0
    print(" Token%30s | %-30sToken" % ("Error", "Correct"))
    print("-" * 38 + "+" + "-" * 38)
    for difftype, test_from, test_to, gold_from, gold_to in matcher.get_opcodes():
        if difftype == "equal":
            true_positives += test_to - test_from
        else:
            false_positives += test_to - test_from
            false_negatives += gold_to - gold_from
            error_chunks += 1
            test_chunk = " ".join(test_tokens[test_from:test_to])
            gold_chunk = " ".join(gold_tokens[gold_from:gold_to])
            print("%6d%30s | %-30s%d" % (test_from, test_chunk, gold_chunk,gold_from))
    precision = 1.0 * true_positives / (true_positives + false_positives)
    recall = 1.0 * true_positives / (true_positives + false_negatives)
    fscore = 2.0 * precision * recall / (precision + recall)
    print()
    print("Test size: %5d tokens" % len(test_tokens))
    print("Gold size: %5d tokens" % len(gold_tokens))
    print("Nr errors: %5d chunks" % error_chunks)
    print("Precision: %5.2f %%" % (100 * precision))
    print("Recall: %5.2f %%" % (100 * recall))
    print("F-score: %5.2f %%" % (100 * fscore))
    

def corpus_length(corpus): 
    """
    Returns the length of the corpus
    """
    corpus_len = len(corpus)
    return corpus_len

def number_types(corpus): 
    """
    Returns the number of types of the corpus
    """
    number_of_types = len(set(corpus))
    return number_of_types


def average_length(corpus):
    """
    Returns the average token length of the corpus
    """
    token_size = 0
    for i in corpus:
        token_size += len(i)
    return token_size/len(corpus)


def longest_token(corpus):
    """
    Returns a tuple with the token with longest length and its length
    """
    token_length = max([(len(x), x) for x in corpus])
    long_token = [(len(i), i) for i in corpus if len(i) == token_length[0]]
    return long_token


def freq_dist(corpus):
    """
    calculate frequency distribution of tokens
    """
    fd = nltk.FreqDist(corpus)
    return fd


def hapaxes(corpus): 
    """
    Returns the number of hapaxes
    """
    fd = freq_dist(corpus)
    length_hapaxes = len(fd.hapaxes()) 
    return length_hapaxes


def percentage(count, total):
    """
    Calculates a percentage
    """
    return 100*count/total


def most_frequent(corpus): 
    """
    Returns the 10 most frequent types of the corpus
    """
    fd = nltk.FreqDist(corpus)
    return fd.most_common(10)


def percentage_common_types (corpus):
    """
    Calculates the percent of the total tokens in the corpus that constist of these types
    """ 
    total = sum([t[1] for t in most_frequent(corpus)])
    return percentage(total, corpus_length(corpus))


def divide_corpus(corpus, number_of_partitions):
    """
    Returns a list that divides the corpus in 10 equally large subcorpora starting in 0
    """
    partition_length = corpus_length(corpus) / number_of_partitions
    list_of_index = []
    for i in range(number_of_partitions + 1):
        list_of_index.append(partition_length*i)
    list_of_index = [int(i) for i in list_of_index]
    ind_bigr = nltk.bigrams(list_of_index)
    corpus_parts = []
    for bigr in ind_bigr:
        corpus_parts.append(corpus[bigr[0]:bigr[1]])
    return corpus_parts


def hapaxes_parts(corpus_parts):
    """
    Returns the number of hapaxes per partition
    """
    hapaxes = [] 
    l = [] 
    for part in corpus_parts:
        fd = freq_dist(part)
        hapaxes.append(fd.hapaxes())
    
    for x in range(len(hapaxes)): 
        p = []
        for hapax in hapaxes[x]:
            if hapax not in l:
                p.append(hapax)
        hapaxes[x] = p
        l.extend(hapaxes[x])

    number_hapaxes = [len(sub_hapaxes) for sub_hapaxes in hapaxes]
    return number_hapaxes


def percentage_hapaxes(corpus_parts, corpus):
    """
    Returns the percentage of hapaxes for every partition
    """
    percentage_h = []
    count = 0
    dv = divide_corpus(corpus, 10)
    hapax_parts = hapaxes_parts(corpus_parts)
    for x in hapax_parts:
        percentage_h.append(percentage(x, len(dv[count])))
        count += 1
    return percentage_h

def plots(corpus_parts, corpus):
    """
    given the data obtained by the functions divide_corpus(corpus, number_of_partitions) and hapaxes_parts(corpus_parts),
    the graphic for the number of hapaxes per partition is plotted
    """
    """
    given the data obtained by the function percentage_hapaxes(dv_corpus, tokenized_corpus),
    the graphic for the percentage of hapaxes per partition is plotted
    """
    h_parts = hapaxes_parts(corpus_parts)
    part_size = [x for x in range(len(h_parts))]
    
    percent_h = percentage_hapaxes(corpus_parts, corpus)
    percent_length = [i for i in range(len(percent_h))]    
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.setp(ax1, xticks=np.arange(0, len(part_size), 1))
    plt.setp(ax2, xticks=np.arange(0, len(percent_length), 1))
    fig.suptitle('Number (left) and percentage (right) of hapaxes in each part')
    ax1.bar(part_size, h_parts)
    ax2.bar(percent_length, percent_h)      
    return plt.show()


def ngram(corpus, n):
    """
    returns the following tuple: (unique ngrams in the corpus, percentage of unique ngrams over the total ngrams).
    The 'n' defines the extension of the ngrams (e.g., bigrams, trigrams...)
    """
    unique_ngrams = []
    n_grams = list(ngrams(corpus, n))
    nr_grams = nltk.FreqDist(n_grams)
    for x in nr_grams:
        if nr_grams[x] == 1:
            unique_ngrams.append(x)
    unique_ngrams = len(unique_ngrams)
    unique_ngrams_percent = percentage(unique_ngrams, len(n_grams))
    return (unique_ngrams, unique_ngrams_percent)


def corpus_statistics(corpus, d_corp):
    """ 
    prints the result of each function in the corpus statistics part
    """
    print('There are {} types of a total of {} tokens in the corpus.\n' .format(number_types(corpus), corpus_length(corpus)))
    print('There average token length is {}.\n' .format(average_length(corpus)))
    print('The longest token is {}.\n' .format(longest_token(corpus)))
    print('The number of hapaxes is {} and represents the {} of the corpus.\n.' .format(hapaxes(corpus), percentage(hapaxes(corpus), corpus_length(corpus))))
    print('The 10 most frequent types of the total tokens are {}  and represent the {}%.\n' .format(most_frequent(corpus), percentage_common_types(corpus))) 
    print('The hapaxes present in each of the 9 partitions are {}.\n' .format(hapaxes_parts(d_corp)))
    print('The percentage of hapaxes for each partition is {}.\n' .format(percentage_hapaxes(d_corp, corpus)))
    plots(d_corp, corpus)
    print('\nIn the tupla {}, the first element is the number of unique bigrams, and the second element is the percentage of unique bigrams from all the bigrams in the corpus. Similarly, in this tupla {}, the first element is the number of unique trigrams, and the second element is the percentage of unique trigrams from all the bigrams in the corpus.' .format(ngram(corpus, 2), ngram(corpus, 3)))      


# command line interpreter:
if __name__ == "__main__":
    nr_files = 199
    corpus_text = get_corpus_text(nr_files)
    gold_tokens = get_gold_tokens(nr_files)
    tokens = tokenize_corpus(corpus_text, pattern)
    evaluate_tokenization(tokens, gold_tokens)
    print("\nCorpus Statistics:\n")
    d_corp = divide_corpus(tokens, 10) 
    corpus_statistics(tokens, d_corp)


