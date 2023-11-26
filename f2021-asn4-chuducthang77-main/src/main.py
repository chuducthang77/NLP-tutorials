# This program receives the tagger type and the path to a test file
# as command line parameters and outputs the POS tagged version of that file.


import argparse
from sklearn.model_selection import KFold
import nltk
from nltk.tag import hmm, brill, brill_trainer, UnigramTagger
from nltk.tbl.template import Template
from nltk.tag.brill import Pos, Word, fntbl37, nltkdemo18, brill24

def main(tagger_type, train, test, output):
    """
    :param tagger_type: Type of tagger
    :param test: Dir of test files
    :param output: Dir of output files
    :param train: Dir of train files
    :return: None
    """
    # process input data
    data = process_data(train, test)
    train_data = data[0]
    test_sent = data[1]
    test_token = data[2]
    # Train the tagger sepcified by the user
    if tagger_type == 'hmm':
        tagger = hmm_tagger(train_data, test_sent)
    elif tagger_type == 'brill':
        tagger = brill_tagger(train_data, test_sent)
    # Run the tagger on the test data and output the results
    test_output(tagger, test_token, test_sent, output)

def process_data(train, test):
    """
    Seperates the data into a list of words
    :param train: Dir of train files
    :param test: Dir of test files
    :return: train_data, test_sent, test_token
    """
    # Process train data
    train_data = []
    with open(train, 'r') as file:
        texts = file.read().splitlines()
        temp = []
        for text in texts:
            if text != '':
                text = tuple(text.split(' '))
                temp.append(text)
            else:
                train_data.append(temp)
                temp = []

    # Process test data
    test_sent = []
    test_token = []
    with open(test, 'r') as file:
        texts = file.read().splitlines()
        temp = []
        temp1 = []
        for text in texts:
            if text != '':
                text = text.split(' ')
                temp.append(tuple(text))
                temp1.append(text[0])
            else:
                test_sent.append(temp)
                test_token.append(temp1)
                temp = []
                temp1 = []
    return [train_data, test_sent, test_token]


def hmm_tagger(train_data, test_sent):
    """
    Trains the HMM tagger
    :param train_data: List of words and their tags from trainig data
    :param test_sent: List of words and their tags from test data
    :return: Trained HMM tagger
    """
    # define possible estimation functions for the HMM tagger
    mle = lambda fd, bins: hmm.MLEProbDist(fd)
    laplace = nltk.LaplaceProbDist
    ele = nltk.ELEProbDist
    bell = nltk.WittenBellProbDist
    def lidstone(gamma):
        return lambda fd, bins: hmm.LidstoneProbDist(fd, gamma, bins)
    lidstone1 = lidstone(0.1)
    lidstone2 = lidstone(0.5)
    lidstone3 = lidstone(1)
    lidstone4 = lidstone(0.01)
    lidstone5 = lidstone(0.02)
    lidstone6 = lidstone(0.04)
    lidstone7 = lidstone(0.06)
    lidstone8 = lidstone(0.08)

    # list of estimators
    estimators = [mle, laplace, ele, bell, lidstone1, lidstone2, lidstone3,
                    lidstone4, lidstone5, lidstone6, lidstone7, lidstone8]
    # corresponding labels for estimators
    names = ['mle', 'laplace', 'ele', 'bell', 'lidstone-0.1',
                'lidstone-0.5', 'lidstone-1', 'lidstone-0.01', 'lidstone-0.02',
                'lidstone-0.04', 'lidstone-0.06', 'lidstone-0.08']
    results = []
    best_acc = 0
    best_estimator = 0
    for i in range(len(estimators)): # for each estimator
        average_acc = 0
        kf = KFold(n_splits=5) # 5-fold cross validation
        for train_index, test_index in kf.split(train_data): # for each fold
            # split training data into train, test and validation sets
            x_train, x_test = [], []
            for j in train_index:
                x_train.append(train_data[j])
            for j in test_index:
                x_test.append(train_data[j])
            # train the tagger using current estimator
            trainer = hmm.HiddenMarkovModelTrainer()
            tagger = trainer.train_supervised(x_train, estimator=estimators[i])
            average_acc += tagger.accuracy(x_test) / 5 # calculate accuracy and average across folds
        results.append(average_acc)
        if average_acc > best_acc: # choose the estimator with the best accuracy
            best_acc = average_acc
            best_estimator = i
    print('Parameter tuning results:')
    print(dict(zip(names, results))) # show results for each estimator
    print('Best estimator: ' + names[best_estimator])

    # train the tagger with the best estimator found above using all the training data
    trainer = hmm.HiddenMarkovModelTrainer()
    tagger = trainer.train_supervised(train_data, estimator=estimators[best_estimator])
    # print accuracy on test data
    print("Accuracy of optimized HMM tagger: " + str(tagger.accuracy(test_sent)))
    return tagger


def brill_tagger(train_data, test_sent):
    """
    Trains the Brill tagger
    :param train_data: List of words and their tags from trainig data
    :param test_sent: List of words and their tags from test data
    :return: Trained Brill tagger
    """
    # regex tagger template for the Brill tagger that we use as backoff
    backoff = nltk.RegexpTagger([
        (r'^-?[0-9]+(\.[0-9]+)?$', 'CD'),
        (r'(The|the|A|a|An|an)$', 'AT'),
        (r'.*able$', 'JJ'),
        (r'.*ness$', 'NN'),
        (r'.*ly$', 'RB'),
        (r'.*s$', 'NNS'),
        (r'.*ing$', 'VBG'),
        (r'.*ed$', 'VBD'),
        (r'.*', 'NN')
    ])

    # define possible rule templates for the Brill tagger
    templates = {
        'template_1': [Template(Pos([-1])), Template(Pos([-1]), Word([0]))],
        'template_2': fntbl37(),
        'template_3': nltkdemo18(),
        'template_4': brill24(),
        'template_5': [
                brill.Template(brill.Pos([-1])),
                brill.Template(brill.Pos([1])),
                brill.Template(brill.Pos([-2])),
                brill.Template(brill.Pos([2])),
                brill.Template(brill.Pos([-2, -1])),
                brill.Template(brill.Pos([1, 2])),
                brill.Template(brill.Pos([-3, -2, -1])),
                brill.Template(brill.Pos([1, 2, 3])),
                brill.Template(brill.Pos([-1]), brill.Pos([1])),
                brill.Template(brill.Word([-1])),
                brill.Template(brill.Word([1])),
                brill.Template(brill.Word([-2])),
                brill.Template(brill.Word([2])),
                brill.Template(brill.Word([-2, -1])),
                brill.Template(brill.Word([1, 2])),
                brill.Template(brill.Word([-3, -2, -1])),
                brill.Template(brill.Word([1, 2, 3])),
                brill.Template(brill.Word([-1]), brill.Word([1])),
            ]
    }
    counts = [5, 20, 100, 250] # possible paramters for max_rules used by the Brill tagger trainer
    results = []
    best_acc = 0
    best_template = None
    best_template_name = None
    best_count = 0
    # labels for each template and count combination
    names = ['template_1_5', 'template_1_20', 'template_1_100', 'template_1_250', 'template_2_5', 'template_2_20',
                'template_2_100', 'template_2_250', 'template_3_5', 'template_3_20', 'template_3_100', 'template_3_250',
                'template_4_5', 'template_4_20', 'template_4_100', 'template_4_250', 'template_5_5', 'template_5_20', 
                'template_5_100', 'template_5_250']
    for template_name, template in templates.items():
        for count in counts:
            average_acc = 0
            kf = KFold(n_splits=5) # 5-fold cross validation
            for train_index, test_index in kf.split(train_data):
                # split training data into train, test and validation sets
                x_train, x_test = [], []
                for j in train_index:
                    x_train.append(train_data[j])
                for j in test_index:
                    x_test.append(train_data[j])
                # train the tagger using the current template and count (parameters)
                baseline = UnigramTagger(x_train, backoff=backoff) # unigram tagger with regex tagger as backoff
                Template._cleartemplates()
                trainer = brill_trainer.BrillTaggerTrainer(baseline, template)
                tagger1 = trainer.train(x_train, max_rules=count)
                average_acc += tagger1.accuracy(x_test) / 5 # calculate accuracy and average across folds
            results.append(average_acc)
            if average_acc > best_acc:
                # choose the template and count with the best accuracy
                best_acc = average_acc
                best_template = template
                best_count = count
                best_template_name = template_name

    print('Parameter tuning results:')
    print(dict(zip(names, results))) # show results for each template and count combination
    print('Best template and max_rules: ' + str(best_template_name) + ' and '+ str(best_count))

    # train the tagger with the best template and count found above using all the training data
    baseline = UnigramTagger(train_data, backoff=backoff)
    Template._cleartemplates()
    trainer = brill_trainer.BrillTaggerTrainer(baseline, best_template)
    tagger = trainer.train(train_data, max_rules=best_count)
    print("Accuracy of optimized Brill tagger: " + str(tagger.accuracy(test_sent)))
    return tagger

def test_output(tagger, test_token, test_sent, output):
    """
    Tests the tagger on the test data and writes the output to a file
    :param tagger: Trained tagger
    :param test_token: List of words from test data without tags
    :param test_sent: List of words and tags from test data
    :param output: Name of the output file
    """
    # Predict the test data
    results = []
    predicted_tag = tagger.tag_sents(test_token)
    for sent in predicted_tag:
        temp2 = []
        for token in list(sent):
            result = ' '.join(token)
            result+='\n'
            temp2.append(result)
        temp2.append('\n')
        results+= temp2

    # Save results to output txt file
    with open(output, 'w') as file:
        file.write(''.join(results))

    confusion_matrix = nltk.ConfusionMatrix(tag_list(test_sent), tag_list(predicted_tag))
    print('Confusion matrix')
    print(confusion_matrix.pretty_format(sort_by_count=True, show_percents=True, truncate=9))
    print('End of confusion matrix')

    print('Precision, Recall, F-measure')
    print(confusion_matrix.evaluate())
def tag_list(tagged_sents):
    """
    :param tagged_sents: list contains words and its corresponding tags of the sentences
    :return: list of only tags of the sentences
    """
    tags = []
    for sent in tagged_sents:
        for (word, tag) in sent:
            tags.append(tag)
    return tags

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Determine whether the sentence is grammatical")
    parser.add_argument("--tagger", type=str, default='hmm', help="Provide tagger type")
    parser.add_argument("--train", type=str, default="data/train.txt", help="Provide path to directory of training data")
    parser.add_argument("--test", type=str, default="data/test.txt", help="Provide path to directory of testing data")
    parser.add_argument("--output", type=str, default="output/test_hmm.txt", help="Provide path to directory of output")

    args = parser.parse_args()
    main(args.tagger, args.train, args.test, args.output)
