# Naive Bayes and Logistic Regression
## UNM CS429
### Vasileios Grigorios Kourakos and Chai Capili, 2023

### Links
* [Github repository](https://github.com/ccapili808/NaiveBayes-Regression/main)
* [Report](https://github.com/ccapili808/NaiveBayes-Regression/blob/main/NaiveBayes_Logistic_Regression.pdf)

### Libraries Used
* [Matrix-toolkits-java](https://github.com/fommil/matrix-toolkits-java/)
* [ojAlgo](https://github.com/optimatika/ojAlgo)

## To Run:
    Our Program runs with the NaiveBayesLogisticRegression.jar file
    Upon running the user will have the following options: 

        [1] Naive Bayes
            Takes in: 
                i) Beta value

        [2] Logistic Regression 
            Takes in: 
                i) Lambda value 
               ii) Learning rate value 
              iii) Number of iterations

        [3] File paths
            There is no option to input file paths while the program is running. To run the jar, place the files in 
            the data set below in the same folder/directory that the jar is in. The file names need to be exactly
            the same as the ones below. 

    The beta, lambda, and eta(learning rate) values need to be a number and the number of iterations
    needs to be an integer.
    Once the user runs one of either LR or NB, the program will run and then report the accuracy. 
    Please allow some time for running.

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
    Our well documented results and analysis of these results can by found in our report. 
    This section will discuss the code which was used in our Naive Bayes class. The naive bayes class is well commented
    and more detailed notes on implementation can be found there as well. 
    
    Data Structures: 
        Naive Bayes doesn't use any external libraries and is coded purely in java. 

        Beta option: A float representation of the Beta option the user selects. 
                * if the user types in "default", beta will be 1 / | V |
                * otherwise, the beta option will correspond to the float the user inputs. 
                * Note: this code will not work properly if the input is not a number or "default", but it will still 
                        run. Please use a valid beta option when running the code. 

        Hashmaps are used to denote different pieces of our data set such as: 

            private HashMap<Integer,Integer> classTotals = new HashMap<>();
            private HashMap<Integer,Integer> totalWords = new HashMap<>();
            private HashMap<Integer, HashMap<Integer,Integer>> wordTotals = new HashMap<>();
            private HashMap<Integer, HashMap<Integer,Double>> wordProbabilities = new HashMap<>();
            private HashMap<Integer,Double> classProbabilities = new HashMap<>();
            private HashMap<Integer, String> vocabulary = new HashMap<>();
        
        These hashmaps allowed for the calculation such as: P(X), P(Y), P(X | Y), and more. 

        Naive Bayes uses MLE to calculate P(Y) which is P(Y_k) = (#docs labeled Y_k / total # docs)
        Naive Bayes calculates P(X | Y) with MAP using two different methods for Beta. 

        When the user selects "default" for beta P(X | Y) is calculated with prior distribution Dirichlet where: 
        (1 + beta, ... 1 + beta) where beta is 1 / | V |. 

        Users can also calculate P(X | Y) with different beta values. Our findings on the discussion between using 
        "default" as Beta vs other beta values is on our report. 

        Our Naive Bayes classfier classifies Y^(new) = max[log_2(P(Y_k)) + Sigma_i (# of X_i^(new))log_2(P(X_i | Yk))]

        After the user gives a beta option the following methods are run sequentially: 
            createDataSet(): Reads in the training set into hashmaps
            calculateProbabilities(): Calculate P(Y) and P(X|Y) using the hashmaps
            mutualInformation(): Ranks the most important words in our dataset. Results in the report. 
            calculateAccuracy(): Calculates the accuracy of the model against the validation set
            After these methods run, the class reads in the testing file and calls predictClass() which predicts the 
            class of a document using the model. 
            
            Finally, the results of the model are printed to predictions.txt

### Logistic Regression
    The accuracy and analysis of results of this model can be found in our report. This section will provide a high 
    level overview of the Logistic Regression class. Further detailed information can be found in the class' code 
    comments. This model can perform ~2-3 iterations of gradient descent per second or ~9000 iterations per hour. 
        
        This class implements multinoomial Logistic Regression and Gradient Descent, which can be described as: 
![image](https://user-images.githubusercontent.com/115299284/227747070-4860eb9f-34b6-4b4e-a395-09bbb93830fc.png)

    
        This implementation features two libraries (listed above) which aided with sparse matrices, dense matrices, and 
        fast matrix operations. 
        The training and validation matrices use sparse matrices, with matrix-toolkits-java greatly improving 
        performance. The rest of the matrices are dense since there are not many zero values. The matrices used in 
        this implementation are as follows: 


        private LinkedSparseMatrix testMatrix = new LinkedSparseMatrix(2000,61189);
        private LinkedSparseMatrix xMatrix = new LinkedSparseMatrix(10000,61189);
        private DenseMatrix classificationsMatrix = new DenseMatrix(12000,1);
        private DenseMatrix weightsMatrix = new DenseMatrix(20,61189);
        private DenseMatrix probabilities = new DenseMatrix(20,10000);
        private DenseMatrix deltaMatrix = new DenseMatrix(20,10000);
        private DenseMatrix lineVector = new DenseMatrix(1,61189);
        private DenseMatrix columnMax = new DenseMatrix(61189, 1);
        private DenseMatrix columnMin = new DenseMatrix(61189, 1);
        private DenseMatrix columnMeans;
        private DenseMatrix columnSD;
        
        Logistic regression stores the following inputs from the user before running:  
            float eta: learning rate
            float lambda: penalty term
            int iterations: the number of iterations (stopping criteria) 
        
        After taking user input, logistic regression then runs the following three methods: 
            1) createDataSet()
                1.1) TfIdfScaling()
                1.2) NormalizeMatrix()
            2) train()
                for # of iterations
                    2.1) calculateProbabilities()
                    2.2) checkAccuracy()
            3) predict()
        
        Detailed explanation of each method are contained in the comments of each method within the Logistic Regression 
        class. Analysis of the results of this model and implementation can be found in the report. Furthermore, while 
        running Logistic Regression, Every iteration, the iteration number, the training data accuracy, testing data 
        accuracy and conditional data likelihood are printed in the console.


# Conclusion 
    Both Naive Bayes and Logistic Regression contain detailed analysis through comments for any confusion in this 
    README. Additionally, the report contains graphs, guided questions, and further analysis on the implementation, 
    requirements, and results. 
    

