import nltk
from nltk.corpus import brown
from prettytable import PrettyTable
nltk.download('brown')
nltk.download('universal_tagset')


def universal_tagged_sents(genre, tagset='universal'): 
    """
    Returns the tagged sentences of the given Brown category.
    """ 
    return nltk.corpus.brown.tagged_sents(categories=genre, tagset='universal')


def brown_tagged_words(genre, tagset='brown'): 
    """
    Returns the tagged sentences of the given Brown category.
    """ 
    return nltk.corpus.brown.tagged_words(categories=genre, tagset='brown')


def print_brown_statistics(list_of_genres):
    """
    Returns some statistics of selected categories of the Brown corpus
    """
    sents_length = []
    words_length = []
    sents_aver = []
    words_aver = []
    tags_U = []
    tags_B = []
    for genre in list_of_genres:
        sentences = universal_tagged_sents(genre, tagset='universal')
        sents_length.append(len(sentences))
        words_U = nltk.corpus.brown.tagged_words(categories=genre, tagset='universal')
        words_length.append(len(words_U))
        words_B = brown_tagged_words(genre, tagset='brown')
        
        count_s = 0
        average = 0
        for s in sentences:
            count_s += len(s)
        average = count_s/sents_length[list_of_genres.index(genre)]
        sents_aver.append(average)
        
        count_w = 0
        tags_1 = []
        for (word, tag) in words_U:
            count_w += len(word)
            tags_1.append(tag)
        average = count_w/words_length[list_of_genres.index(genre)]
        words_aver.append(average)
        total_tags = len(set(tags_1))
        tags_U.append(total_tags)
        
        tags_2 = []
        for (word, tag) in words_B:
            tags_2.append(tag)
        total_tags = len(set(tags_2))
        tags_B.append(total_tags)
    return (tags_U, tags_B, sents_length, words_length, sents_aver, words_aver)


def part1():
    """
    prints the results of part 1 in a table
    """
    print("\nPart 1")
    genres = ["fiction", "government", "news", "reviews"]
    b_s = print_brown_statistics(genres)
    brown_statistics = PrettyTable()
    brown_statistics.field_names = ["Brown genre", "Uni POS tags", "Orig POS tags", "Sentences", "Words", "Sent. length", "Word length"]
    
    for g in range(len(genres)):
        brown_statistics.add_row([genres[g], b_s[0][g], b_s[1][g], b_s[2][g], b_s[3][g], round(b_s[4][g], 2), round(b_s[5][g], 2)])  
 
    print(brown_statistics)


def common_ngrams(corpus, n, n_results):
    """
    returns both the frequency and the accumulated frequency of the n most common ngrams in the 'news' genre corpus
    """
    fd = nltk.FreqDist()
    results = []
    for s in corpus:
        tags = [tag for (word, tag) in s]
        ngrams = nltk.ngrams(tags, n, pad_left=True, pad_right=True, left_pad_symbol="$", right_pad_symbol="$")
        for ngram in ngrams:
            fd[ngram] += 1
    total_ngrams = fd.N()
    m_common = fd.most_common(n_results)
    accum_freq = 0
    for item in m_common:
        freq = item[1]*100/total_ngrams
        accum_freq += freq
        results.append((item[0], freq, accum_freq))
    return results


def part2():
    """
    prints the results of part 2 in a table 
    """
    print("\nPart 2")
    
    corpus = nltk.corpus.brown.tagged_sents(categories = 'news', tagset = 'universal')
    uni_tagger = common_ngrams(corpus, 1, 10)
    unigram_table = PrettyTable()
    unigram_table.field_names = ["1-gram", "Frequency", "Accum.freq."]
    for i in range(len(uni_tagger)):
        unigram_table.add_row([uni_tagger[i][0], round(uni_tagger[i][1], 2), round(uni_tagger[i][2], 2)])
    print('Unigram statistics')
    print(unigram_table)
    
    
    bi_tagger = common_ngrams(corpus, 2, 10)
    bigram_table = PrettyTable()
    bigram_table.field_names = ["2-gram", "Frequency", "Accum.freq."]    
    for i in range(len(bi_tagger)):
        bigram_table.add_row([bi_tagger[i][0], round(bi_tagger[i][1], 2), round(bi_tagger[i][2], 2)]) 
    print('\nBigram statistics')
    print(bigram_table)

    
    tri_tagger = common_ngrams(corpus, 3, 10)
    trigram_table = PrettyTable()

    trigram_table.field_names = ["3-gram", "Frequency", "Accum.freq."]    
    for i in range(len(tri_tagger)):
        trigram_table.add_row([tri_tagger[i][0], round(tri_tagger[i][1], 2), round(tri_tagger[i][2], 2)])
    print('\nTrigram statistics')
    print(trigram_table)


def split_corpus(genre):
    """
    given a genre, the corpus is splitted in two: train and test 
    """
    sents = brown.tagged_sents(categories=genre, tagset='universal') 
    test = sents[:500]
    training = sents[500:]
    return (training, test)


def most_frequent_tag(data):
    """
    given a set of tagged sentences it extracts the most frequent tag
    input expected: list of sentences (list of tuples)
    """
    tags = []
    for x in data:
        for (word, tag) in x:
            tags.append(tag)
    
    return nltk.FreqDist(tags).max()


def train_nltk_taggers(train, test):
    """
    trains and avaluate a bigram tagger (with backoff cascade)
    input expected: train and test data (list of sentences (list of tuples))
    """
    defaultTagger = nltk.DefaultTagger(most_frequent_tag(train))
    affixTagger = nltk.AffixTagger(train, backoff=defaultTagger)
    uniTagger = nltk.UnigramTagger(train, backoff=affixTagger)
    biTagger = nltk.BigramTagger(train, backoff=uniTagger)
    triTagger = nltk.TrigramTagger(train, backoff=biTagger)
    
    return (defaultTagger, affixTagger, uniTagger, biTagger, triTagger)


def evaluate_taggers(train, test):
    """
    returns the accuracy and errors for a given train and test corpus
    """
    taggers = train_nltk_taggers(train, test)
    accuracy = []
    for tag in taggers:
        accuracy.append(tag.evaluate(test))
    
    errors = []
    for acc in accuracy:
        errors.append(1/(1-acc))

    return (accuracy, errors)


def part3():
    """
    prints the results of part 3 in a table
    """
    print("\nPart 3")
    train, test = split_corpus('news')
    accuracy = evaluate_taggers(train, test)
    acc = PrettyTable()
    taggers = ['default', 'affix', 'unigram', 'bigram', 'trigram']
    acc.field_names = ["Genre: news", "Accuracy", "Errors"]
    
    for i in range(len(taggers)):
        acc.add_row([taggers[i], round(accuracy[0][i]*100, 2), round(accuracy[1][i], 2)])    
    
    print(acc)


def bigram_tagger(train, test):
    """
    returns the accuracy and errors of the biTagger for any (tran, test) tuple 
    """
    return (evaluate_taggers(train, test)[0][3], evaluate_taggers(train, test)[1][3])


def test_on_training_set(train, test):
    """
    returns accuracy and errors for different train and test corpus using the biTagger
    """
    train_test = bigram_tagger(train, test)
    train_train = bigram_tagger(train, train)
    
    return (train_test, train_train)


def test_different_genres(genres):
    """
    returns accuracy and errors using the biTagger trained with the news genre and tested with all the genres
    """
    accuracy_errors = []
    train = split_corpus('news')[0]
    for genre in genres:
        test = split_corpus(genre)[1]
        accuracy_errors.append(bigram_tagger(train, test))
    
    return accuracy_errors


def train_different_sizes(sizes):
    """
    returns accuracy and errors using the biTagger tested in different percentages of the total corpus size
    """
    accuracy_errors = []
    for size in sizes:
        train, test = split_corpus('news')
        size = int(size*len(train)/100) 
        train_size = train[0:size]
        accuracy_errors.append(bigram_tagger(train_size, test))
    
    return accuracy_errors


def compare_train_test_partitions(genre):
    """
    returns accuracy and errors using the biTagger in different partitions
    """
    train, test = split_corpus(genre)
    new_sents = brown.tagged_sents(categories=genre, tagset='universal')
    new_train = new_sents[:-500]
    new_test = new_sents[-500:]
    accuracy_errors = bigram_tagger(train, test)
    new_accuracy_errors = bigram_tagger(new_train, new_test)
    
    return [accuracy_errors, new_accuracy_errors]


def splitting_training_testing(data):
    """
    splits the input data in the tuple (train, test)
    """
    train = data[500:]
    test = data[:500]
    
    return (train, test)


def dic_simple_tags():
    """
    dictionary mapping from  information
    """
    dic = nltk.Index([('NOUN', 'N'), ('NUM', 'N'), ('ADJ', 'NP'), ('DET', 'NP') ,('PRON', 'NP'), ('VERB', 'V'), ('ADP', 'AUX'), ('ADV', 'AUX'), ('CONJ', 'AUX'), ('PRT', 'AUX'), ('X', 'AUX'), ('.', 'DELIM')])
    
    return dic


def super_simple_tag(sentsU):
    """
    given a data set tagged with the universal tagset, 
    returns a data set with a supersimple tagset
    """
    dic = dic_simple_tags()
    newCorpus = []
    for s in sentsU:
        newSent = []
        for t in s:
            newSent.append((t[0], dic[t[1]][0]))
        newCorpus.append(newSent)
    
    return newCorpus


def setting_data(genre):
    """
    given a brown genre, extracts and prepares the training and testing data (to build taggers)
    outputs three data sets (three tuples), containing each:
        - train set
        - testing set
    """
    newsB = brown.tagged_sents(categories=genre, tagset='brown')
    trainB, testB = splitting_training_testing(newsB)
    newsU = brown.tagged_sents(categories=genre, tagset='universal')
    trainU, testU = splitting_training_testing(newsU)
    newsS = super_simple_tag(newsU)
    trainS, testS = splitting_training_testing(newsS)
    
    return (trainB, trainU, trainS, testB, testU, testS)


def compare_different_tagsets(genre):
    """
    returns the accuracy, errors and number of tags using three tagsets: unviersal, brown and super_simple_tag
    """
    trainB, testB = setting_data(genre)[0], setting_data(genre)[3]
    trainU, testU = setting_data(genre)[1], setting_data(genre)[4]
    trainS, testS = setting_data(genre)[2], setting_data(genre)[5]
    accuracy_U = bigram_tagger(trainU, testU)
    accuracy_S = bigram_tagger(trainS, testS)
    accuracy_B = bigram_tagger(trainB, testB)

    U_sents = brown.tagged_sents(categories=genre, tagset='universal')
    S_sents = super_simple_tag(U_sents)
    B_sents = brown.tagged_sents(categories=genre, tagset='brown')
    sents = [U_sents, S_sents, B_sents]
    total_tags = []
    for x in range(len(sents)):
        tags = []
        for s in sents[x]:
            for (word, tag) in s: 
                tags.append(tag)
        total_tags.append(len(set(tags)))
    
    return (accuracy_U, accuracy_S, accuracy_B, total_tags)


def part4():
    """
    prints the results of part 4 in a table
    """
    print("\nPart 4")
    train, test = split_corpus('news')
    part_4a = test_on_training_set(train, test)
    p4a = PrettyTable()
    p4a.field_names = ["Train sentences", "Accuracy", "Errors", "Test sentences"]
    column = ['news-train', 'news-test', 'news-train', 'news-train']
    
    for i in range(len(part_4a)):
        p4a.add_row([column[i], round(part_4a[i][0]*100, 2), round(part_4a[i][1], 2), column[i]])
        
    print(p4a)
    
    genres = ["fiction", "government", "news", "reviews"]
    part_4b = test_different_genres(genres)
    p4b = PrettyTable()
    p4b.field_names = ["Train sentences", "Accuracy", "Errors", "Test sentences"]
    column = ["fiction-test", "government-test", "news-test", "reviews-test"]
    
    for i in range(len(part_4b)):
        p4b.add_row(["news-train", round(part_4b[i][0]*100, 2), round(part_4b[i][1],2), column[i]])
        
    print(p4b)
    
    sizes = [100, 75, 50, 25]
    part_4c = train_different_sizes(sizes)
    p4c = PrettyTable()
    p4c.field_names = ["Train sentences", "Accuracy", "Errors", "Test sentences"]
    column = ["news-train (100%)", "news-train (75%)", "news-train (50%)", "news-train (25%)"]
    
    for i in range(len(part_4c)):
        p4c.add_row([column[i], round(part_4c[i][0]*100, 2), round(part_4c[i][1],2), 'news-test'])
        
    print(p4c)
    
    part_4d = compare_train_test_partitions('news')
    p4d = PrettyTable()
    p4d.field_names = ["Genre", "Accuracy", "Errors", "Partition"]
    column = ["test = news[:500], train = news[500:]", "test = news[-500:], train = news[:-500]"]
    
    for i in range(len(part_4d)):
        p4d.add_row(['news', round(part_4d[i][0]*100, 2), round(part_4d[i][1],2), column[i]])
        
    print(p4d)
    
    part_4e = compare_different_tagsets('news')
    p4e = PrettyTable()
    p4e.field_names = ["Tagset", "Accuracy", "Errors", "Nr.tags"]
    tagset = ["news universal", "news super-simple", "news original"]
    
    for i in range(len(part_4e) - 1):
        p4e.add_row([tagset[i], round(part_4e[i][0]*100, 2), round(part_4e[i][1],2), part_4e[3][i]])
        
    print(p4e)


if __name__ == "__main__":
    parts = [part1(), part2(), part3(), part4()]
    for p in parts: p
       
