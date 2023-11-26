# This program should loop over the files listed in the input file directory,
# assign perplexity with each language model,
# and produce the output as described in the assignment description.
import csv
import argparse
import os
import numpy as np
from collections import Counter

np.seterr(divide = 'ignore')
class UnsmoothedModel():
    def __init__(self, name=None, text=None):
        self.text = text  # text we are using the model on.
        self.name = name  # name of text file
        self.n = 1  # n parameter
        self.unk_threshold = 1  # threshold for replacing words with <UNK>
        self.distribution = None  # distribution of n-grams
        self.vocab = {}  # vocabulary
        self.len = None  # lenght of token list

    def train(self):
        # Tokenize character level
        tokens = list(self.text)
        token_counter = Counter(tokens)
        # Create vocabulary
        for token in token_counter.keys():
            # if a word appears less than unk_threshold times, replace it with <UNK>
            if token_counter[token] < self.unk_threshold:
                if "<UNK>" in self.vocab:
                    self.vocab["<UNK>"] += token_counter[token]
                else:
                    self.vocab["<UNK>"] = token_counter[token]
            else:
                self.vocab[token] = token_counter[token]

                # Replace token with less than threshold with <UNK>
        tokens = [x if x in self.vocab else "<UNK>" for x in tokens]

        # Create n-gram and n_1-gram for character
        ngram = []  # list of n-grams
        for i in range(len(tokens) - (self.n - 1)):
            j_gram = []  # current n-gram
            for j in range(self.n):
                j_gram.append(tokens[i + j])
            ngram.append(tuple(j_gram))
        if self.n > 1: 
            n_1_gram = [] # list of n-1-grams
            for i in range(len(tokens) - (self.n - 2)):
                j_gram = []  # current n-1-gram
                for j in range(self.n - 1):
                    j_gram.append(tokens[i + j])
                n_1_gram.append(tuple(j_gram))

        self.len = len(tokens)
        self.distribution = Counter(ngram)
        if self.n > 1:
            self.char_counter = Counter(n_1_gram)
        else:
            self.char_counter = self.vocab

    def ngram_probability(self, ngram, n_1_gram):
        # get conditional probability
        # if no distribution, return -inf
        if n_1_gram is not None:
            char_count = self.char_counter[n_1_gram]
            if char_count != 0:
                count_1 = char_count
            else:
                return float('-inf')
        else:
            count_1 = self.len
        if self.distribution[ngram] != 0:
            count = self.distribution[ngram]
        else:
            return float('-inf')

        log_prob = np.log2(count / count_1)
        return log_prob

    def perplexity(self, text):
        # Tokenize and replace any unknown words with <UNK>
        tokens = [x if x in self.vocab else "<UNK>" for x in list(text)]

        # Create ngrams
        ngram = []
        for i in range(len(tokens) - (self.n - 1)):
            j_gram = []
            for j in range(self.n):
                j_gram.append(tokens[i + j])
            ngram.append(tuple(j_gram))
        if self.n > 1:
            n_1_gram = []
            for i in range(len(tokens) - (self.n - 2)):
                j_gram = []
                for j in range(self.n - 1):
                    j_gram.append(tokens[i + j])
                n_1_gram.append(tuple(j_gram))

        # add up n-gram probabilities to get log probability
        log_p = 0
        for i in range(len(ngram)):
            if self.n > 1:
                log_p += self.ngram_probability(ngram[i], n_1_gram[i])
            else:
                log_p += self.ngram_probability(ngram[i], None)
        # return perplexity. 
        return np.power(2, -((1 / len(tokens)) * log_p))

class LaplaceModel():
    def __init__(self, name=None, text = None):
        self.n = 3  # n parameter
        self.text = text  # text we are using the model on.
        self.name = name  # name of text file
        self.distribution = None  # distribution of n-grams
        self.vocab = {}  # vocabulary
        self.len = None  # lenght of token list

    def train(self):
        tokens = list(self.text)
        token_counter = Counter(tokens)
        # Create vocabulary
        for token in token_counter.keys():
            self.vocab[token] = token_counter[token]

            # Replace token with less than threshold with <UNK>
        tokens = [x if x in self.vocab else "<UNK>" for x in tokens]

        # Create n-gram and n_1-gram for character
        ngram = []  # list of n-grams
        for i in range(len(tokens) - (self.n - 1)):
            j_gram = []  # current n-gram
            for j in range(self.n):
                j_gram.append(tokens[i + j])
            ngram.append(tuple(j_gram))
        if self.n > 1:
            n_1_gram = []  # list of n-1 grams
            for i in range(len(tokens) - (self.n - 2)):
                j_gram = []  # current n-1 gram
                for j in range(self.n - 1):
                    j_gram.append(tokens[i + j])
                n_1_gram.append(tuple(j_gram))

        self.len = len(tokens)
        self.distribution = Counter(ngram)
        if self.n > 1:
            self.char_counter = Counter(n_1_gram)
        else:
            self.char_counter = self.vocab

    def ngram_probability(self, ngram, n_1_gram):
        if n_1_gram is not None:
            char_occurence = self.char_counter[n_1_gram]
            if char_occurence != 0:
                count_1 = char_occurence + len(self.vocab.keys())  # adjust denom for smoothing
            else:
                count_1 = len(self.vocab.keys())
        else:
            count_1 = self.len + len(self.vocab.keys())
        if self.distribution[ngram] != 0:
            count = self.distribution[ngram]
        else:  # add one smoothing
            count = 1

        return(np.log2(count / count_1))

    def perplexity(self, text):
        tokens = list(text)

        # Create ngrams
        ngram = []
        for i in range(len(tokens) - (self.n - 1)):
            j_gram = []
            for j in range(self.n):
                j_gram.append(tokens[i + j])
            ngram.append(tuple(j_gram))
        if self.n > 1:
            n_1_gram = []
            for i in range(len(tokens) - (self.n - 2)):
                j_gram = []
                for j in range(self.n - 1):
                    j_gram.append(tokens[i + j])
                n_1_gram.append(tuple(j_gram))

        # add up n-gram probabilities to get log probability
        log_prob = 0
        for i in range(len(ngram)):
            if self.n > 1:
                log_prob += self.ngram_probability(ngram[i], n_1_gram[i])
            else:
                log_prob += self.ngram_probability(ngram[i], None)
        # return perplexity
        return np.power(2, -((1 / len(tokens)) * log_prob))

class InterpolationModel():
    def __init__(self, name=None, text=None):
        self.n = 3  # n parameter
        self.multi_grams = [None for i in range(self.n)]  # list of n-grams
        self.multi_grams_lambda = [None for i in range(self.n)]  # list n-grams for adjusting weights
        self.weights = []  # list of weights
        self.unk_threshold = 1  # threshold for replacing words with <UNK>
        self.text = text  # text we are using the model on.
        self.name = name  # name of text file
        self.vocab = {}  # vocabulary
        self.len = None  # lenght of token list

    # helper function to create n-grams given characters and n.
    def create_n_gram(self, tokens, order):
        ngram = []
        for i in range(len(tokens) - (order - 1)):
            j_gram = []
            for j in range(order):
                j_gram.append(tokens[i + j])
            ngram.append(tuple(j_gram))
        return ngram

    def train(self):
        tokens = list(self.text)
        token_counter = Counter(tokens)
        # Create vocabulary
        for token in token_counter.keys():
            if token_counter[token] < self.unk_threshold:
                if "<UNK>" in self.vocab:
                    self.vocab["<UNK>"] += token_counter[token]
                else:
                    self.vocab["<UNK>"] = token_counter[token]
            else:
                self.vocab[token] = token_counter[token]

                # Replace token with less than threshold with <UNK>
        tokens = [x if x in self.vocab else "<UNK>" for x in tokens]

        # Seperate the training data in two sets to prevent contamination
        # Lambda values determined based on the lambda_token
        # the rest is used to to calculate perplexity
        lambda_token = tokens[3500:]
        tokens = tokens[:3500]

        # Create n-gram and n_1-gram for characters
        self.len = len(tokens)
        for i in range(len(self.multi_grams)):
            self.multi_grams[i] = Counter(self.create_n_gram(tokens, self.n - i))

        # Same but on set used in deleted interpolation
        for i in range(len(self.multi_grams_lambda)):
            self.multi_grams_lambda[i] = Counter(self.create_n_gram(lambda_token, self.n - i))
        self.deleted_interpolation()

    def deleted_interpolation(self):
        """
        This function is used to calculate the weights for the interpolation model.
        Follows the algorithm described in chapter 8 of textbook.
        """
        # initialize weights
        weights = np.zeros(self.n)
        # ngram of largest n
        highest_n_gram = self.multi_grams_lambda[0]

        for gram in highest_n_gram:
            prob = []
            count = 0
            for i in range(self.n):
                gr = gram[i:]
                count = self.multi_grams_lambda[i][gr]
                if i == 0:
                    count = count
                if i != self.n - 1:
                    count_1 = self.multi_grams_lambda[i + 1][gr[:-1]]
                else:
                    count_1 = self.len
                # find probablity of each n-gram
                prob.append((count - 1) / (count_1 - 1) if count_1 - 1 != 0 else 0)
            # choose index to maximize likelihood of corpus
            ind = prob.index(max(prob))
            # increment weight of index
            weights[ind] += count

        self.weights = [np.divide(w, np.sum(weights)) for w in reversed(weights)]

    def perplexity(self, text):
        char_tokens = [x if x in self.vocab else "<UNK>" for x in list(text)]
        n_grams = self.create_n_gram(char_tokens, self.n)
        log_prob = 0
        for token in n_grams:
            prob = []
            for i in range(self.n):
                gr = token[i:]
                count = self.multi_grams[i][gr]
                if i != self.n - 1:
                    count_1 = self.multi_grams[i + 1][gr[:-1]]
                else:
                    count_1 = self.len
                # calculate probability of each n-gram
                prob.append(count / count_1 if count_1 != 0 else 0)

            # apply weights to each probability to get weighted probability.
            # sum weighted probabilities to get log probability
            s = 0
            for i in range(len(self.weights)):
                s += self.weights[i] * prob[i]
            log_prob += np.log2(s)
        # return perplexity
        return np.power(2, -((1 / len(char_tokens)) * log_prob))


def main(train, test, output, model_type):
    """
    :param train: Dir of train files
    :param test: Dir of test files
    :param output: Dir of output files
    :param model_type: Dir of smoothing type
    :return: None
    """

    train_files = os.listdir(train)
    train_files.sort()

    results = [['Testing_file', 'Training_file', 'Perplexity', 'n']]

    # Load and tokenize train files
    models = []
    for file in train_files:
        with open(train + '/' + file, 'r') as f:
            contents = f.read()
            if model_type == 'laplace':
                model = LaplaceModel(file, contents)
            elif model_type == 'unsmoothed':
                model = UnsmoothedModel(file, contents)
            else:
                model = InterpolationModel(file, contents)
            #  train model
            model.train()
            models.append(model)

    # Load and tokenize test files
    # select language match with lowest perplexity for each language
    test_files = os.listdir(test)
    test_files.sort()
    for file in test_files:
        best_perplexity = float('inf')
        best_model = None
        best_n = None
        with open(test + '/' + file, 'r') as f:
            test_contents = f.read()
            for model in models:
                per = model.perplexity(test_contents)
                if per < best_perplexity:
                    best_perplexity = per
                    best_model = model.name
                    best_n = model.n
        results.append([file, best_model, best_perplexity, best_n])

    # Save results to csv file
    with open(output, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Determine whether the sentence is grammatical")
    parser.add_argument("train", type=str, default="data/train", help="Provide path to directory of training data")
    parser.add_argument("test", type=str, default="data/dev", help="Provide path to directory of testing data")
    parser.add_argument("output", type=str, default="output/results_dev_unsmoothed.csv", help="Provide path to directory of output")
    parser.add_argument("--laplace", nargs='?', default='laplace')
    parser.add_argument("--unsmoothed", nargs='?', default='unsmoothed')
    parser.add_argument("--interpolation", nargs='?', default='interpolation')

    args = parser.parse_args()
    if args.laplace == None:
        model_type = 'laplace'
    elif args.interpolation == None:
        model_type = 'interpolation'
    else:
        model_type = 'unsmoothed'
    main(args.train, args.test, args.output, model_type)
