from tkinter.filedialog import test
import numpy as np
import pandas as pd
from string import punctuation
import argparse
from collections import Counter
from sklearn.model_selection import KFold
import nltk

class NaiveBayes:
    def __init__(self, train_path, test_path, output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.output_path = output_path
        self.bags = [[],[],[],[]] # performer-0, director-1, publisher-2, characters-3
        self.probs = [{},{},{},{}] # performer-0, director-1, publisher-2, characters-3
        self.class_count = np.zeros(4) # performer-0, director-1, publisher-2, characters-3
        self.class_prob = np.zeros(4)
        self.train_size = 0
        self.classes = ['performer', 'director', 'publisher', 'characters']

    def preprocess_data(self):
        """
        Preprocess the data
        """
        # tokenize words and add them in appropriate list
        # head (and tail) is added as one word
       
        stopwords_en = ["'m","'s",'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']

        stopwords_en_withpunct = set(stopwords_en).union(set(punctuation)) # set of stopwords and punctuation

        all_words = []
        
        df = pd.read_csv(self.train_path)
        sentences = df.to_numpy() # convert dataframe to numpy 2d array
        for sentence in sentences: # for each sentence
            label = sentence[2]
            head = sentence[3].split()
            tail = sentence[4].split()
            words = sentence[1].split()
            i = 0

            # select right bow
            if label == 'performer':
                bow = self.bags[0]
                self.class_count[0] += 1
            elif label == 'director':
                bow = self.bags[1]
                self.class_count[1] += 1
            elif label == 'publisher':
                bow = self.bags[2]
                self.class_count[2] += 1
            elif label == 'characters':
                bow = self.bags[3]
                self.class_count[3] += 1

            temp = ""
            for word in words:
                word = word.lower() # convert to lowercase
                if str(i) in head: # if word is in head keep in temp until end of head
                    temp += word + " "
                elif str(i) in tail: # if word is in tail keep in temp until end of tail
                    temp += word + " "
                elif word in stopwords_en_withpunct or word.isdigit(): # if word is a stopword or punctuation or digit, ignore
                    if temp != "":
                        bow.append(temp)
                        all_words.append(temp)
                        temp = ""
                else:
                    if temp != "": # if temp is not empty add it to bow
                        bow.append(temp)
                        all_words.append(temp)
                        temp = ""
                    bow.append(word) # add word to bow
                    all_words.append(word)
                i += 1
        for bow in self.bags:
            for word in all_words:
                bow.append(word) # add all words to each bow (for smoothing)

    def get_probs(self):
        self.train_size = np.sum(self.class_count)
        self.class_prob = self.class_count / self.train_size # normalize class count
        self.class_prob = np.log(self.class_prob) # take log of class probability

        self.bags = [np.array(bow) for bow in self.bags] # convert list to numpy array

        # calculate the probability of each word in each class (P(word|class))
        i = 0
        for bow in self.bags:
            p_c_e = Counter(bow)
            for key in p_c_e:
                p_c_e[key] = np.log(p_c_e[key] / len(bow))
            self.probs[i] = p_c_e
            i+=1


    def predict(self):
        """
        Predict class of each sentence in test data
        """
        df = pd.read_csv(self.test_path)
        sentences = df.to_numpy()

        stopwords_en = ["'m","'s",'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']

        stopwords_en_withpunct = set(stopwords_en).union(set(punctuation)) # set of stopwords and punctuation

        predictions = []
        accurate = 0

        for sentence in sentences:
            sent_id = sentence[0]
            correct_label = sentence[2]
            words = sentence[1].split()
            head = sentence[3].split()
            tail = sentence[4].split()

            best_prob = -np.inf
            best_class = None
            index = 0
            for bow in self.probs:
                
                current_prob = self.class_prob[self.probs.index(bow)] # start with log class probability
                i = 0
                temp = ""
                for word in words:
                    word = word.lower() # convert to lowercase
                    if str(i) in head: # if word is in head keep in temp until end of head
                        temp += word + " "
                    elif str(i) in tail: # if word is in tail keep in temp until end of tail
                        temp += word + " "
                    elif word in stopwords_en_withpunct or word.isdigit(): # if word is a stopword or punctuation or digit, ignore
                        if temp != "":
                            try :
                                prob = bow[temp]
                                current_prob += prob
                            except: # if word is not in bow, ignore
                                pass
                            temp = ""
                    else:
                        if temp != "":
                            try :
                                prob = bow[temp]
                                current_prob += prob
                            except: # if word is not in bow, ignore
                                pass
                            temp = ""
                        try :
                            prob = bow[word]
                            current_prob += prob
                        except: # if word is not in bow, ignore
                            pass
                    i += 1
                if current_prob > best_prob:  # if current probability is greater than best probability, update best probability and best class
                    best_prob = current_prob
                    best_class = index
                index += 1
            best_class = self.classes[best_class]
            if best_class == correct_label:
                accurate += 1
            predictions.append(str(correct_label)+', '+str(best_class)+', '+str(sent_id))

        accuracy = accurate/len(sentences)  
        # Save results to output txt file
        if self.output_path:
            print("Test Accuracy: ", accuracy)
            file = open(self.output_path, 'w')
            file.write('original_label, output_label, row_id\n')
            for i in range(len(predictions)):
                file.write(predictions[i]+'\n')
        return accuracy
        


def main(train_path, test_path, output_path):
    """
    Main function
    param train_path: path to the training data
    param test_path: path to the test data
    param output_path: path to the output file
    """
    # train classifier on traing data using 3-fold cross validation and report accuracy.
    train_temp_path = 'data/cross_val_train.csv'
    test_temp_path = 'data/cross_val_test.csv'
    train_data = open(train_path, 'r')
    train_data = train_data.readlines()
    header = train_data.pop(0)
    kf = KFold(n_splits=3)
    accuracies = []
    for train_index, test_index in kf.split(train_data):
        # for every fold, create training and test sets according to shuffle and run the classifier using those files.
        train_temp = open(train_temp_path, "w")
        train_temp.write(header)
        for i in train_index:
            train_temp.write(train_data[i]) 
        test_temp = open(test_temp_path, "w")
        test_temp.write(header)
        for i in test_index:
            test_temp.write(train_data[i])
        classifier = NaiveBayes(train_temp_path, test_temp_path, None)
        classifier.preprocess_data()
        classifier.get_probs()
        accuracy = classifier.predict()
        accuracies.append(accuracy)
    # average training accuracies over folds
    train_accuracy = 0
    for accuracy in accuracies:
        train_accuracy += accuracy
    print("Training accuracy using 3-fold cross validation: "+str(train_accuracy/3))

    # train classifier on entire training set, measure accuracy on test set and output results.
    classifier = NaiveBayes(train_path, test_path, output_path)
    classifier.preprocess_data()
    classifier.get_probs()
    classifier.predict()

    print_confusion_matrix(output_path)
    
def print_confusion_matrix(output_path):
    correct = []
    predicted = []
    df = pd.read_csv(output_path)
    results = df.to_numpy()
    for result in results:
        correct.append(result[0])
        predicted.append(result[1][1:])
    
    cm = nltk.ConfusionMatrix(correct, predicted)
    print()
    print('Confusion matrix')
    print(cm.pretty_format(sort_by_count=True))
    print()
    print('Precision, Recall, F-measure')
    print(cm.evaluate())
    print()
    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Naive bayes classifier")
    parser.add_argument("--train", type=str, default="data/train.txt", help="Provide path to directory of training data")
    parser.add_argument("--test", type=str, default="data/test.txt", help="Provide path to directory of testing data")
    parser.add_argument("--output", type=str, default="output/test.txt", help="Provide path to directory of output")

    args = parser.parse_args()
    main(args.train, args.test, args.output)