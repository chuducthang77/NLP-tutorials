# Intro to NLP - Assignment 4

## Team
|Student name| CCID  |
|------------|-------|
|student 1   | thang |
|student 2   | hmarcand |

Please note that CCID is **different** from your student number.

## TODOs

In this file you **must**:
- [X] Fill out the team table above. 
- [X] Make sure you submitted the URL on eClass.
- [X] Acknowledge all resources consulted (discussions, texts, urls, etc.) while working on an assignment.
- [X] Provide clear installation and execution instructions that TAs must follow to execute your code.
- [X] List where and why you used 3rd-party libraries.
- [X] Delete the line that doesn't apply to you in the Acknowledgement section.

## Acknowledgement 
In accordance with the UofA Code of Student Behaviour, we acknowledge that  

- We have listed all external resources we consulted for this assignment.

### Ressources

 * Probability distributions provided by nltk. We used them as estimators the hmm tagger: https://www.nltk.org/howto/probability.html
 * We followed the following example to get started with the brill tagger: https://www.nltk.org/api/nltk.tag.brill_trainer.html 

## 3-rd Party Libraries

* `main.py L:[109] and L[199]:` used `sklearn.model_selection.KFold` to shuffle and seperate the data to perform k fold cross validation.

## Installation
We reccommend running the program in a virtual environment. To install dependencies:
```
pip install -r requirements.txt
```

## Execution
Example usage: use the following command in the current directory.

`python3 src/main.py --tagger hmm --train data/train.txt --test data/test.txt --output output/test_hmm.txt`

where `--tagger` can take `hmm` or `brill` as arguments

## Data

The assignment's training data can be found in [data/train.txt](data/train.txt), the in-domain test data can be found in [data/test.txt](data/test.txt), and the out-of-domain test data can be found in [data/test_ood.txt](data/test_ood.txt).
