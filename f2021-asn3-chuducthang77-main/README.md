# Intro to NLP - Assignment 3

## Team
|Student name| CCID  |
|------------|-------|
|student 1   | thang |
|student 2   | hmarcand |

## TODOs

In this file you **must**:
- [X] Fill out the team table above. Please note that CCID is **different** from your student number.
- [X] Acknowledge all resources consulted (discussions, texts, urls, etc.) while working on an assignment. Non-detailed oral discussion with others is permitted as long as any such discussion is summarized and acknowledged by all parties.
- [X] Provide clear installation and execution instructions that TAs must follow to execute your code.

## Installation
We reccommend running the program in a virtual environment. To install dependencies:
```
pip install -r requirements.txt
```

## Execution
Example usage: use the following command in the current directory.

`python3 src/main.py data/train/ data/dev/ output/results_dev_laplace.csv --laplace`

The option --'model' is used to define what model to use. model can be one of: unsmoothed, laplace, interpolation

## Data

The assignment's training data can be found in [data/train](data/train) and the development data can be found in [data/dev](data/dev).


## References
Divide by zero grams:
https://stackoverflow.com/questions/28867242/float-division-by-zero-error-related-to-ngram-and-nltk

Log Perplexity formula:
https://towardsdatascience.com/perplexity-in-language-models-87a196019a94

Discussed high level interpolation implementation with Akrash Sharma.
  
Deleted Interpolation Algorithm:
chapter 8 of *An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition*, by Daniel Jurafsky and James H. Martin
