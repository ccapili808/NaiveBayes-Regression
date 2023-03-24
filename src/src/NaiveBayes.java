import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.*;

/**
 * This class trains the Naive Bayes model for the 20newsgroups dataset, then
 * predicts the labels of the testing set documents.
 * The user can specify the value of beta to use, which is the strength or the prior.
 * Additionally, it implements mutual information to print out the best 100 words,
 * the accuracy against the validation set (we used 10000 documents for training and
 * 2000 documents for validation). These are not randomized as it seemed like the examples are already
 * pretty randomized in the file, and I wanted to get consistent training results
 * for testing and debugging. The confusion matrix for the validation set is also printed after training.
 * This class just used hashmaps since there are no iterations and the operations
 * just need to be performed once. Performance could probably be improved but
 * is acceptable for the purposes of this project.
 */
public class NaiveBayes {
    //set the names of the input files
    private String vocabularyFile = "vocabulary.txt";
    private String trainingFile = "training.csv";
    private String classificationFile = "newsgrouplabels.txt";
    private String testingFile = "testing.csv";
    private String betaOption;
    //create structures for the data we need, mostly hashmaps were used
    private HashMap<Integer,Integer> classTotals = new HashMap<>();
    private HashMap<Integer,Integer> totalWords = new HashMap<>();
    private HashMap<Integer, HashMap<Integer,Integer>> wordTotals = new HashMap<>();
    private HashMap<Integer, HashMap<Integer,Double>> wordProbabilities = new HashMap<>();
    private HashMap<Integer,Double> classProbabilities = new HashMap<>();
    private HashMap<Integer, String> vocabulary = new HashMap<>();
    private ArrayList<String[]> validationSet = new ArrayList<>();
    //this 2d array is used for mutual information
    private int[][] xOccurances = new int[61188][20];
    //confusion matrix 2d array
    private int[][] confusionMatrix = new int [20][20];

    /**
     * Constructor for the NaiveBayes class
     * @param beta the beta value that the model should use
     * @throws FileNotFoundException
     * @throws UnsupportedEncodingException
     */
    public NaiveBayes (String beta) throws FileNotFoundException, UnsupportedEncodingException {
        betaOption = beta;
        //read training set into hashmaps
        createDataSet();
        //calculate P(Y) and P(X|Y)
        calculateProbabilities();
        //calculate word mutual information
        mutualInformation();
        //calculate the accuracy of the model against the validation set
        calculateAccuracy();
        //start predicting and write predictions to txt file
        System.out.println("Reading testing file and generating predictions...");
        Scanner sc = new Scanner(new File(testingFile));
        //print the predictions to a txt file in the correct Kaggle format
        PrintWriter writer = new PrintWriter("predictions.txt", "UTF-8");
        writer.println("id,class");
        while(sc.hasNextLine()) {
            String[] document = sc.nextLine().split(",");
            writer.println(document[0] + "," + predictClass(document));
        }
        writer.close();
    }

    /**
     * This method reads in the training file and puts the values in the correct hashmaps/arrays
     * @throws FileNotFoundException
     */
    public void createDataSet() throws FileNotFoundException {
        System.out.println("Reading training set...");
        Scanner sc = null;
        sc = new Scanner(new File(trainingFile));
        //read every line in training set and build the data needed to calculate Bayes terms
        //10000 training set
        for (int k = 0; k < 10000; k++) {
            String[] line = sc.nextLine().split(",");
            int classification = Integer.parseInt(line[line.length-1]);
            //add class totals
            if(classTotals.containsKey(classification)) {
                classTotals.put(classification, classTotals.get(classification)+1);
            }
            else {
                classTotals.put(Integer.parseInt(line[line.length-1]), 1);
            }
            for (int i=1; i < (line.length-1); i++) {
                int wordCount = Integer.parseInt(line[i]);
                if (wordCount>0) {
                    //add 1 to occurance of word for the specific class if word count is greater than zero
                    xOccurances[i - 1][classification - 1] += 1;
                }
                //add specific word count from each document to the total count of that word for the class
                if (wordTotals.containsKey(i)) {
                    if (wordTotals.get(i).containsKey(classification)) {
                        wordTotals.get(i).put(classification, wordTotals.get(i).get(classification) + wordCount);
                    } else {
                        wordTotals.get(i).put(classification, wordCount);
                    }
                } else {
                    HashMap<Integer, Integer> wordHashMap = new HashMap<>();
                    wordHashMap.put(classification, wordCount);
                    wordTotals.put(i, wordHashMap);
                }
                //add to class's total word count
                if (totalWords.containsKey(classification)) {
                    totalWords.put(classification, totalWords.get(classification) + wordCount);
                } else {
                    totalWords.put(classification, wordCount);
                }
            }
        }
        //add the other 2000 documents to the validation set
        for (int k = 10000; k < 12000; k++) {
            String[] line = sc.nextLine().split(",");
            validationSet.add(line);
        }
    }

    /**
     * This method calculates P(Y) and P(X|Y) for all the training data
     */
    public void calculateProbabilities() {
        System.out.println("Calculating P(Y) and P(X|Y) for every class and word...");
        //calculate P(Y) for each class
        int totalDocuments = 0;
        for (int i:classTotals.values()
        ) {
            totalDocuments+= i;
        }
        for (int i = 1; i < 21; i++) {
            double classProbability = (double)classTotals.get(i)/(double)totalDocuments;
            classProbabilities.put(i,classProbability);
        }
        //set beta to what the user chose, "default" means 1/V where V is the vocabulary size
        double beta;
        if (betaOption.equals("default")) {
            beta = (double) 1 / (double) wordTotals.keySet().size();
        }
        else {
            beta = Double.parseDouble(betaOption);
        }
        //calculate P(X|Y) for every word and class
        for (int j = 0; j < 20; j++) {
            double denominator = (totalWords.get(j+1) + (beta * wordTotals.keySet().size()));
            for (int i = 0; i < wordTotals.keySet().size(); i++) {
                double probability = (((double) wordTotals.get(i+1).get(j+1)) + beta)/denominator;
                if(wordProbabilities.containsKey(i+1)) {
                    wordProbabilities.get(i + 1).put(j + 1, probability);
                }
                else {
                    HashMap<Integer,Double> wordProbability = new HashMap<>();
                    wordProbability.put(j + 1, probability);
                    wordProbabilities.put(i+1,wordProbability);
                }
            }
        }
    }

    /**
     * This method calculates the mutual information of words
     * and prints out the 100 words with the highest MI
     */
    public void mutualInformation() {
        System.out.println("Calculating mutual information of words...");
        //use beta for smoothing in case there is division by zero/log of zero
        double beta;
        if (betaOption.equals("default")) {
            beta = (double) 1 / (double) wordTotals.keySet().size();
        }
        else {
            beta = Double.parseDouble(betaOption);
        }
        //read in the words from the vocabulary
        Scanner sc = null;
        try {
            sc = new Scanner(new File(vocabularyFile));
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
        for (int i = 0; i < 61188; i++) {
            String word = sc.nextLine();
            vocabulary.put(i+1,word);
        }
        HashMap<Double, Integer> MI = new HashMap<>();

        // for each word calculate the following:
        // p(x,y)*log(p(x,y)/(p(x)*p(y)))
        // do this for combinations of x,y (0,0) (0,1) (1,0) (1,1)
        // 1) x word not occuring in classes that are not y
        // 2) x word not occuring in class y
        // 3) x word occuring in classes that are not y
        // 4) x word occuring in class y
        // sum those 4 metrics, multiply by the probability of class y
        // finally sum the above metric from all classes to get the average MI of the word

        for (int k = 0; k < 61188;k++) {
            double wordMI = 0;
            double Px = 0;
            //get the x occurances for each class
            for (int i =0 ; i<20; i++) {
                Px += xOccurances[k][i];
            }

            //calculate the sums described above
            for (int j = 0; j < 20; j++) {
                double tempMI = 0;
                double notOccurInOtherClasses = 0;
                for (int q = 0;q<20;q++) {
                    notOccurInOtherClasses += (classTotals.get(q+1) - xOccurances[k][q]);
                }
                double occurInOtherClasses = 0;
                for (int q = 0;q<20;q++) {
                    if (q != j) {
                        occurInOtherClasses += xOccurances[k][q];
                    }
                }
                notOccurInOtherClasses += -(classTotals.get(j+1) - xOccurances[k][j]);
                for (int i = 0; i < 4; i++) {
                    if (i==0) {
                        double Pij = ((double)(notOccurInOtherClasses/10000));
                        double Probx = (10000-Px)/10000;
                        tempMI += Pij * log2((Pij+beta)/((Probx*(1-classProbabilities.get(j+1)))+beta));
                    }
                    else if (i==1){
                        double Pij = (double)(classTotals.get(j+1)- xOccurances[k][j])/10000;
                        double Probx = (double)(10000-Px)/10000;
                        tempMI += Pij * log2((Pij+beta)/((Probx*(classProbabilities.get(j+1)))+beta));
                    }
                    else if (i==2){
                        double Pij = occurInOtherClasses/10000;
                        double Probx = (double)Px/10000;
                        tempMI += Pij * log2((Pij+beta)/((Probx*(1-classProbabilities.get(j+1)))+beta));
                    }
                    else if (i==3){
                        double Pij = (double)xOccurances[k][j]/10000;
                        double Probx = (double)Px/10000;
                        tempMI += Pij * log2((Pij+beta)/((Probx*classProbabilities.get(j+1))+beta));
                    }
                }
                wordMI += classProbabilities.get(j+1) * tempMI;
            }
            MI.put(wordMI, k+1);
        }
        //sort the list from max to min
        ArrayList<Double> sortedMI = new ArrayList<>(MI.keySet());
        Collections.sort(sortedMI);
        Collections.reverse(sortedMI);
        //print out the 100 best words
        for (int i = 0;i < 100; i++) {
            int wordIndex = MI.get(sortedMI.get(i));
            System.out.println(vocabulary.get(wordIndex));
        }
    }

    /**
     * This method calculates the model's accuracy against the validation set and
     * prints the confusion matrix in console.
     */
    public void calculateAccuracy () {
        System.out.println("Calculating validation set accuracy and confusion matrix...");
        double correctPredictions = 0;
        for (String[] validationInstance: validationSet
             ) {
            HashMap<Double, Integer> classProb = new HashMap<>();
            //calculate Y for each class given all the words of a document
            for (int j = 1; j < 21; j++) {
                double probSum = 0;
                //use log to change to addition instead of multiplication
                for (int i = 1; i < validationInstance.length - 1; i++) {
                    probSum += Integer.parseInt(validationInstance[i]) * log2(wordProbabilities.get(i).get(j));
                }
                classProb.put((log2(classProbabilities.get(j)) + probSum), j);
            }
            //get the max Y from all classes which will be our prediction and return it
            double argmax = Collections.max(classProb.keySet());
            int prediction = classProb.get(argmax);
            int classification = Integer.parseInt(validationInstance[validationInstance.length-1]);
            if (prediction == classification) {
                correctPredictions += 1;
            }
            //add the prediction to the confusion matrix
            confusionMatrix[classification-1][prediction-1]++;
        }
        //calculate accuracy and print out accuracy/confusion matrix
        double accurracy = correctPredictions/2000;
        System.out.println("Validation Set Accuracy: " + accurracy);
        System.out.println("Confusion Matrix:");
        for (int i = 0; i < 20; i++) {
            System.out.println(Arrays.toString(confusionMatrix[i]));
        }
    }

    /**
     * This method predicts the class of a document with the NB model
     * @param document the String[] of the document to predict. contains the values for every word in the document
     * @return the class prediction
     */
    public int predictClass(String[] document) {
        HashMap<Double, Integer> classProb = new HashMap<>();
        //calculate Y for each class given all the words of a document
        for (int j = 1; j < 21; j++) {
            double probSum = 0;
            //use log to change to addition instead of multiplication
            for (int i = 1; i < document.length; i++) {
                probSum += Integer.parseInt(document[i]) * log2(wordProbabilities.get(i).get(j));
            }
            classProb.put((log2(classProbabilities.get(j)) + probSum), j);
        }
        //get the max Y from all classes which will be our prediction and return it
        double argmax = Collections.max(classProb.keySet());
        int index = classProb.get(argmax);
        System.out.println(document[0] + "," + index);
        return index;
    }

    /**
     * This method calculates the log base 2 of a double
     * @param N the double to calculate the log of
     * @return the log
     */
    public static double log2(double N)
    {
        double result = (Math.log(N) / Math.log(2));

        return result;
    }
}
