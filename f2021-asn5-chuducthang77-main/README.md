# Intro to NLP - Assignment 5

## Team
|Student name| CCID |
|------------|------|
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
(**delete the line that doesn't apply to you**)

- We have listed all external resources we consulted for this assignment.
### References
 * Using stopwords: https://www.kaggle.com/code/alvations/basic-nlp-with-nltk/notebook#Stopwords

 Non-detailed oral discussion with others is permitted as long as any such discussion is summarized and acknowledged by all parties.

## 3-rd Party Libraries
You do not need to list `nltk` and `pandas` here.

* `main.py L:190` used `scikit-learn` for creating folds for k-fold cross-validation.
* `main.py L:17, 18, 87` used `numpy` for creating arrays. `main.py L:83, 85, 94` used `numpy` for math operations log and sum.

## Installation
We reccommend running the program in a virtual environment. To install dependencies:
```
pip install -r requirements.txt
```

## Execution
Example usage: use the following command in the current directory.

`python3 src/main.py --train data/train.csv --test data/test.csv --output output/test.csv`

## Data

The assignment's training data can be found in [data/train.txt](data/train.txt),and the test data can be found in [data/test.txt](data/test.txt).
