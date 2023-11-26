import pandas as pd
import argparse
import nltk
from sklearn.metrics import confusion_matrix

def main(input, grammar, output):

    # Load the input file, grammar file and create a parser
    file_input = pd.read_csv(input,sep='\t')
    sentences = file_input.pos
    grammar = nltk.data.load(grammar)
    # parser from notebook in class notes
    parser = nltk.chart.BottomUpLeftCornerChartParser(grammar)

    # make output file
    file_output = pd.DataFrame(columns=['id','ground_truth','prediction'])
    file_output['id'] = file_input['id']
    file_output['ground_truth'] = file_input['label']
    # Determine whether the sentence is grammatical
    for i in range(len(sentences)):
        sent = sentences[i].split()
        try:
            check = False
            for tree in parser.parse(sent):
                check = True
            if check:
                file_output['prediction'][i] = 0  # accept
            else:
                file_output['prediction'][i] = 1  # reject (no parse)
        except:  # if something goes wrong, default to not part of grammar
            file_output['prediction'][i] = 1

    file_output = file_output.astype({'prediction': 'int64'})

    # Calculate precision and recall
    TP, FN, FP, TN = confusion_matrix(file_output['ground_truth'], file_output['prediction']).ravel()
    print("TP = " + str(TP))
    print("FN = " + str(FN))
    print("FP = " + str(FP))
    print("TN = " + str(TN))
    precision = TP/ (TP + FP)
    recall = TP/(TP + FN)
    print('Precision: ', precision)
    print('Recall: ', recall)

    # Save the tsv file, make csv to tsv conversion
    file_output.to_csv(output, sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Determine whether the sentence is grammatical")
    parser.add_argument("input", type=str, default="data/train.tsv", help="Provide path to directory of data input")
    parser.add_argument("grammar", type=str, default="grammars/toy.cfg", help="Provide path to directory of grammar rules")
    parser.add_argument("output", type=str, default="output/train.tsv", help="Provide path to directory of output")

    args = parser.parse_args()
    main(args.input, args.grammar, args.output)
