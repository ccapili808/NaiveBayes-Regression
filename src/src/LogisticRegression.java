import org.ojalgo.data.DataProcessors;
import org.ojalgo.function.PrimitiveFunction;
import org.ojalgo.matrix.store.ElementsSupplier;
import org.ojalgo.matrix.store.Primitive32Store;
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

    private Primitive32Store xMatrix = Primitive32Store.FACTORY.make(12000,61189);
    private Primitive32Store classificationsMatrix = Primitive32Store.FACTORY.make(12000, 1);
    private Primitive32Store probabilities = Primitive32Store.FACTORY.make(20,12000);
    private Primitive32Store weightsMatrix = Primitive32Store.FACTORY.make(20,61189);
    private Primitive32Store deltaMatrix = Primitive32Store.FACTORY.make(20,12000);
    private Primitive32Store xTranspose = Primitive32Store.FACTORY.make(61189,12000);
    PrimitiveFunction.Unary matrixOperation = arg -> {
        if (Math.exp(arg) > 20 || Float.isInfinite((float)Math.exp(arg))) {
            return 20;
        }
        else return Math.exp(arg);
    };


    //the learning rate
    private float eta;

    //the penalty term
    private float lambda;

    //the number of iterations
    private final int ITERATIONS = 10;


    public LogisticRegression(float lambda){
        this.eta = 0.001f;
        this.lambda = lambda;
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
                weightsMatrix.set(i,j,((0.001f*r.nextFloat())));
            }
        }
        System.out.println("Reading training set...");
        Scanner sc = null;
        sc = new Scanner(new File(trainingFile));
        //read every line in training set and build the data needed
        //while (sc.hasNextLine()) {
        for(int l = 0; l<12000;l++) {
            String[] line = sc.nextLine().split(",");
            float classification = Float.parseFloat(line[line.length-1]);
            int documentID = Integer.parseInt(line[0]);
            //print out the document number to see where the code is at
            if(documentID%100 == 0) {
                System.out.println("Reading DocumentID: " + documentID);
            }
            for (int j = 0; j<20; j++) {
                //set the delta matrix
                if (classification == j) {
                    deltaMatrix.set(j,documentID-1,1f);
                }
                else {
                    deltaMatrix.set(j,documentID-1,0.0f);
                }
            }
            //add the document's classification to the matrix
            classificationsMatrix.set(documentID-1,0,classification);
            //set x0 to 1
            xMatrix.set(documentID-1,0,1f);
            for (int i=1; i < (line.length-2); i++) {
                float wordCount = Float.parseFloat(line[i]);
                //add specific word count from each document to the total count of that word to the example matrix
                if(wordCount>0) {
                    xMatrix.set(documentID-1,i,wordCount);
                }
            }
        }
        //get the transpose of the X matrix and make it into a SparseStore for efficient multiplication
        ElementsSupplier temp1 = xMatrix.transpose();
        temp1.supplyTo(xTranspose);

    }

    public void train(){
        //iterate and update weight matrix every time
        for(int k = 0; k<= ITERATIONS; k++){
            System.out.println("Iteration Number: " + k);
            //calculate the new probability matrix using the weights
            calculateProbabilities();

            //perform matrix operations using ojAlgo java library

            ElementsSupplier temp1;
            ElementsSupplier temp2;
            ElementsSupplier temp3;
            ElementsSupplier temp4;
            Primitive32Store matrix1 = Primitive32Store.FACTORY.make(20,12000);
            Primitive32Store matrix2 = Primitive32Store.FACTORY.make(20,61189);

            temp1 = deltaMatrix.onMatching(SUBTRACT, probabilities);
            temp1.supplyTo(matrix1);
            temp2 = matrix1.multiply(xMatrix);
            temp3 = weightsMatrix.onAll(MULTIPLY, lambda);
            temp3.supplyTo(matrix2);
            temp4 = temp2.onMatching(SUBTRACT,matrix2);
            temp4.onAll(MULTIPLY, eta);
            temp4.onMatching(ADD, weightsMatrix);
            temp4.supplyTo(weightsMatrix);
            weightsMatrix.modifyAny(DataProcessors.CENTER_AND_SCALE);

        }
    }

    public void calculateProbabilities() {

        ElementsSupplier temp;
        //multiply weights by x transpose
        temp = weightsMatrix.multiply(xTranspose);
        //make every element e^i

        temp.supplyTo(probabilities);
        for(int i = 0; i < 12000; i++) {
            probabilities.set(19,i,0);
        }
        probabilities.modifyAny(DataProcessors.SCALE);
        probabilities.onAll(matrixOperation);

        //normalize probability columns
        for(int i = 0;i < 12000; i++) {
            probabilities.set(19,i,1f);
            float total = 0.0f;
            for (int j = 0; j < 20; j++) {
                total += probabilities.get(j,i);
            }
            if (total != 0.0f) {
                for (int j = 0; j < 20; j++) {
                    probabilities.set(j, i, (float) (probabilities.get(j, i) / total));
                }
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

    public void setLambda(float lambda) {
        this.lambda = lambda;
    }

}