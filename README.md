# Naive Bayes and Logistic Regression
## UNM CS429
### Vasileios Grigorios Kourakos and Chai Capili, 2023

### Links
* [Github repository](https://github.com/ccapili808/NaiveBayes-Regression)
* [Report]()

## To Run:
    Our Program runs with the NaiveBayesLogisticRegression.jar file
    Upon running the user will have the following options: 

        [1] Naive Bayes
            Takes in: 
                i) beta value

        [2] Logistic Regression 
            Takes in: 
                i) Lambda value 
               ii) Learning rate value 

        [3] File paths
            Options for following files: 
                i) vocabulary
               ii) labels
              iii) training
               iv) testing

    Once the user runs one of either LR or NB, the program will run and then report the accuracy. 
    Please allow some time for running. 

## Libraries and Data Structures
    Our code is able to take in CSV files and apply Naive Bayes and Logistic Regression using Java and the following libraries: 
* [Library One]()
* [Library Two]()

    
    After the CSV file is parsed, these libraries allow us to store various columns and rows as matrices and apply various matrix operations.
    //TODO: 
        discuss what we do w/ matrices and what each library does 

## Code Analysis
### Testing and Training Data
    [vocabulary.txt] is a list of the words that may appear in documents. The line number is word’s d in other ﬁles. 
    That is, the ﬁrst word (’archive’) has wordId 1, the second (’name’) has wordId 2, etc.

    [newsgrouplabels.txt] is a list of newsgroups from which a document may have come. 
    Again, the line number corresponds to the label’s id, which is used in the .label ﬁles. 
    The ﬁrst line (’alt.atheism’) has id 1, etc.

    [training.csv] Speciﬁes the counts for each of the words used in each of the documents. 
    Each line contains 61190 elements. 
    The first element is the document id, the elements 2 to 61189 are word counts for a vectorized representation 
    of words (refer to the vocabulary for a mapping). The last element is the class of the document (20 different classes). 
    All word/document pairs that do not appear in the ﬁle have count 0.

    [testing.csv] The same as training.csv except that it does not contain the last element.
### Naive Bayes
    Reported Accuracy: 
    
    Code discussion: 

### Logistic Regression
    Reported Accuracy: 

    Code discussion: 

## Misc. 
    
# Conclusion 
    Indepth analysis on our findings can be found in our report (link above). 
    

