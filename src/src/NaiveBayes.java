import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.Collections;
import java.util.HashMap;
import java.util.Scanner;

public class NaiveBayes {
    private String vocabularyFile = "vocabulary.txt";
    private String trainingFile = "training.csv";
    private String classificationFile = "newsgrouplabels.txt";
    private String testingFile = "testing.csv";
    private String betaOption;
    private HashMap<Integer,Integer> classTotals = new HashMap<>();
    private HashMap<Integer,Integer> totalWords = new HashMap<>();
    private HashMap<Integer, HashMap<Integer,Integer>> wordTotals = new HashMap<>();
    private HashMap<Integer, HashMap<Integer,Double>> wordProbabilities = new HashMap<>();
    private HashMap<Integer,Double> classProbabilities = new HashMap<>();

    public NaiveBayes (String beta) throws FileNotFoundException, UnsupportedEncodingException {
        betaOption = beta;
        //read training set into hashmaps
        createDataSet();
        //calculate P(Y) and P(X|Y)
        calculateProbabilities();
        //start predicting and write predictions to txt file
        System.out.println("Reading testing file and generating predictions...");
        Scanner sc = new Scanner(new File(testingFile));
        PrintWriter writer = new PrintWriter("predictions.txt", "UTF-8");
        writer.println("id,class");
        while(sc.hasNextLine()) {
            String[] document = sc.nextLine().split(",");
            writer.println(document[0] + "," + predictClass(document));
        }
        writer.close();
    }

    public void createDataSet() throws FileNotFoundException {
        System.out.println("Reading training set...");
        Scanner sc = null;
        sc = new Scanner(new File(trainingFile));
        //read every line in training set and build the data needed to calculate Bayes terms
        while (sc.hasNextLine()) {
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
                //add specific word count from each document to the total count of that word for the class
                if (wordCount >= 0) {
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
                    }
                    else {
                        totalWords.put(classification,wordCount);
                    }
                }
            }
        }
    }

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
        //set beta to what the user chose
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
        /*
        for (int i = 0; i < wordTotals.keySet().size(); i++) {
            HashMap<Integer,Double> wordProbability = new HashMap<>();
            for (int j = 0; j < 20; j++) {
                double probability = (((double) wordTotals.get(i+1).get(j+1)) + beta)/(totalWords.get(j+1) + (beta * wordTotals.keySet().size()));
                wordProbability.put(j+1, probability);
            }
            wordProbabilities.put(i+1, wordProbability);
        }

         */
    }

    public int predictClass(String[] document) {
        HashMap<Double, Integer> classProb = new HashMap<>();
        //calculate Y for each class given all the words of a document
        for (int j = 1; j < 21; j++) {
            double probSum = 0;
            //use log to change to addition instead of multiplication
            for (int i = 1; i < document.length - 1; i++) {
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

    public static double log2(double N)
    {
        double result = (Math.log(N) / Math.log(2));

        return result;
    }
}
