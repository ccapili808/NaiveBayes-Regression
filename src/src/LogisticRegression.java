import org.ojalgo.matrix.store.ElementsSupplier;
import org.ojalgo.matrix.store.Primitive64Store;
import org.ojalgo.matrix.store.SparseStore;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;

import static org.ojalgo.function.constant.PrimitiveMath.*;


public class LogisticRegression {
    private String vocabularyFile = "vocabulary.txt";
    private String trainingFile = "training.csv";
    private String classificationFile = "newsgrouplabels.txt";
    private String testingFile = "testing.csv";

    private Primitive64Store xMatrix = Primitive64Store.FACTORY.make(12000,61189);
    private Primitive64Store classificationsMatrix = Primitive64Store.FACTORY.make(12000, 1);
    private Primitive64Store probabilities = Primitive64Store.FACTORY.make(20,12000);
    private Primitive64Store weightsMatrix = Primitive64Store.FACTORY.make(20,61189);
    private Primitive64Store deltaMatrix = Primitive64Store.FACTORY.make(20,12000);
    private SparseStore<Double> xTranspose;


    //the learning rate
    private double eta;

    //the penalty term
    private double lambda;

    //the number of iterations
    private final int ITERATIONS = 10;


    public LogisticRegression(double lambda){
        this.eta = 0.001;
        this.lambda = lambda;
        try {
            createDataSet();
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
        train();
    }

    public LogisticRegression() {
        eta = 0.001;
        lambda = 0.01;
        try {
            createDataSet();
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
        train();
    }

    public void createDataSet() throws FileNotFoundException {
        Random r = new Random();
        //initialize random weights
        for(int i =0; i <20;i++) {
            for(int j = 0; j<61189;j++) {
               weightsMatrix.set(i,j,(0+(0.1*r.nextDouble())));
            }
        }
        System.out.println("Reading training set...");
        Scanner sc = null;
        sc = new Scanner(new File(trainingFile));
        //read every line in training set and build the data needed
        while (sc.hasNextLine()) {
            String[] line = sc.nextLine().split(",");
            int classification = Integer.parseInt(line[line.length-1]);
            int documentID = Integer.parseInt(line[0]);
            //print out the document number to see where the code is at
            if(documentID%100 == 0) {
                System.out.println("Reading DocumentID: " + documentID);
            }
            for (int j = 0; j<20; j++) {
                //set the delta matrix
                if (classification == j) {
                    deltaMatrix.set(j,documentID-1,1);
                }
            }
            //add the document's classification to the matrix
            classificationsMatrix.set(documentID-1,0,classification);
            //set x0 to 1
            xMatrix.set(documentID-1,0,1);
            for (int i=1; i < (line.length-2); i++) {
                double wordCount = Double.parseDouble(line[i]);
                //add specific word count from each document to the total count of that word to the example matrix
                if(wordCount>0) {
                    xMatrix.set(documentID-1,i,wordCount);
                }
            }
        }
        //get the transpose of the X matrix and make it into a SparseStore for efficient multiplication
        xTranspose = SparseStore.PRIMITIVE64.make(xMatrix.transpose());
    }

    public void train(){
        //iterate and update weight matrix every time
        for(int k = 0; k<= ITERATIONS; k++){
            System.out.println("Iteration Number: " + k);
            //calculate the new probability matrix using the weights
            calculateProbabilities();

            //perform matrix operations using ojAlgo java library

            ElementsSupplier<Double> temp1;
            ElementsSupplier<Double> temp2;
            ElementsSupplier<Double> temp3;
            ElementsSupplier<Double> temp4;
            ElementsSupplier<Double> temp5;
            Primitive64Store matrix1 = Primitive64Store.FACTORY.make(20,12000);
            Primitive64Store matrix2 = Primitive64Store.FACTORY.make(20,61189);

            temp1 = deltaMatrix.onMatching(SUBTRACT, probabilities);
            temp1.supplyTo(matrix1);
            temp2 = matrix1.multiply(xMatrix);
            temp3 = weightsMatrix.onAll(MULTIPLY, lambda);
            temp3.supplyTo(matrix2);
            temp4 = temp2.onMatching(SUBTRACT,matrix2);
            temp5 = temp4.onAll(MULTIPLY, eta);
            temp5.onMatching(ADD, weightsMatrix);
            temp5.supplyTo(weightsMatrix);
        }
    }

    public void calculateProbabilities() {
        ElementsSupplier temp;
        //multiply weights by x transpose
        temp = weightsMatrix.multiply(xTranspose);
        //make every element e^i
        temp.onAll(EXP);
        //put into probabilities matrix
        temp.supplyTo(probabilities);
        //normalize probability columns
        for(int i = 0;i < 12000; i++) {
            probabilities.set(19,i,1);
            double total = 0;
            for (int j = 0; j < 20; j++) {
                total += probabilities.get(j,i);
            }
            for (int j = 0; j < 20; j++) {
                probabilities.set(j,i, (probabilities.get(j,i)/total));
            }
        }
    }

    public void setVocabularyFile(String vocabularyFile) {
        this.vocabularyFile = vocabularyFile;
    }

    public void setTrainingFile(String trainingFile) {
        this.trainingFile = trainingFile;
    }

    public void setClassificationFile(String classificationFile) {
        this.classificationFile = classificationFile;
    }

    public void setTestingFile(String testingFile) {
        this.testingFile = testingFile;
    }

    public void setLambda(double lambda) {
        this.lambda = lambda;
    }

}