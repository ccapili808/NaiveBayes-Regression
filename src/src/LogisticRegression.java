import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.sparse.LinkedSparseMatrix;
import org.ojalgo.array.Array2D;
import org.ojalgo.data.DataProcessors;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;
import static no.uib.cipr.matrix.Matrices.getArray;
import static org.ojalgo.function.constant.PrimitiveMath.*;


public class LogisticRegression {
    private String vocabularyFile = "vocabulary.txt";
    private String trainingFile = "training.csv";
    private String classificationFile = "newsgrouplabels.txt";
    private String testingFile = "testing.csv";

    private LinkedSparseMatrix xMatrix = new LinkedSparseMatrix(12000,61189);
    private DenseMatrix classificationsMatrix = new DenseMatrix(12000,1);
    private DenseMatrix weightsMatrix = new DenseMatrix(20,61189);
    private DenseMatrix probabilities = new DenseMatrix(20,12000);
    private DenseMatrix deltaMatrix = new DenseMatrix(20,12000);


    //the learning rate
    private float eta;

    //the penalty term
    private float lambda;

    //the number of iterations
    private final int ITERATIONS = 10000;


    public LogisticRegression(float lambda){
        this.eta = 0.01f;
        this.lambda = lambda;
        try {
            createDataSet();
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
        train();
        predict();
    }

    public void createDataSet() throws FileNotFoundException {
        Random r = new Random();
        //initialize random weights
        for(int i =0; i <20;i++) {
            for(int j = 0; j<61189;j++) {
                weightsMatrix.set(i,j,((1f*r.nextFloat())));
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
    }

    public void train(){
        //iterate and update weight matrix every time
        for(int k = 0; k<= ITERATIONS; k++){
            System.out.println("Iteration Number: " + k);
            //calculate the new probability matrix using the weights
            calculateProbabilities();

            //update weights
            DenseMatrix temp;
            DenseMatrix temp2 = new DenseMatrix(61189,20);
            DenseMatrix temp3 = new DenseMatrix(20,61189);
            temp = deltaMatrix.copy();
            temp.add(-1, probabilities);
            xMatrix.transABmult(temp, temp2);
            temp2.transpose(temp3);
            temp3.add(-lambda, weightsMatrix);
            weightsMatrix.add(eta, temp3);
            Array2D<Double> array2D = Array2D.R064.rows(getArray(weightsMatrix));
            array2D.modifyAny(DataProcessors.CENTER_AND_SCALE);
            weightsMatrix = new DenseMatrix(array2D.toRawCopy2D());
        }
    }

    public void predict(){
        Scanner sc = null;
        try {
            sc = new Scanner(new File(testingFile));
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
        xMatrix = null;
        xMatrix = new LinkedSparseMatrix(6774,61189);
        probabilities = new DenseMatrix(20,6774);
        //read every line in testing set and build the data needed
        //while (sc.hasNextLine()) {
        for(int l = 0; l<6774;l++) {
            String[] line = sc.nextLine().split(",");
            float classification = Float.parseFloat(line[line.length-1]);
            int documentID = Integer.parseInt(line[0]);
            //print out the document number to see where the code is at
            if(documentID%100 == 0) {
                System.out.println("Reading DocumentID: " + documentID);
            }
            //set x0 to 1
            xMatrix.set(documentID-12001,0,1f);
            for (int i=1; i < (line.length-2); i++) {
                float wordCount = Float.parseFloat(line[i]);
                //add specific word count from each document to the total count of that word to the example matrix
                xMatrix.set(documentID-12001,i,wordCount);
            }
        }

        DenseMatrix temp = new DenseMatrix(6774,20);

        xMatrix.transBmult(weightsMatrix,temp);
        temp.transpose(probabilities);
        for(int i = 0; i < 6774; i++) {
            probabilities.set(19,i,0);
        }
        Array2D<Double> array2D = Array2D.R064.rows(getArray(probabilities));
        array2D.modifyAny(DataProcessors.CENTER_AND_SCALE);
        array2D.modifyAll(EXP);
        probabilities = new DenseMatrix(array2D.toRawCopy2D());

        //normalize probability columns
        for(int i = 0;i < 6774; i++) {
            double total = 0;
            for (int j = 0; j < 20; j++) {
                total += probabilities.get(j,i);
            }
            if (total != 0) {
                for (int j = 0; j < 20; j++) {
                    probabilities.set(j, i, (probabilities.get(j, i) / total));
                }
            }
        }

        for(int i = 0;i < 6774; i++) {
            double argmax = 0;
            int prediction = 0;
            for (int j = 0;j < 20; j++) {
                if (probabilities.get(j,i) > argmax) {
                    argmax = probabilities.get(j,i);
                    prediction = j+1;
                }
            }
            System.out.println(""+(12001+i)+","+prediction);
        }
    }

    public void calculateProbabilities() {

        DenseMatrix temp = new DenseMatrix(12000,20);

        xMatrix.transBmult(weightsMatrix,temp);
        temp.transpose(probabilities);
        for(int i = 0; i < 12000; i++) {
            probabilities.set(19,i,0);
        }


        Array2D<Double> array2D = Array2D.R064.rows(getArray(probabilities));
        array2D.modifyAny(DataProcessors.CENTER_AND_SCALE);
        array2D.modifyAll(EXP);

        probabilities = new DenseMatrix(array2D.toRawCopy2D());



        //normalize probability columns
        for(int i = 0;i < 12000; i++) {
            double total = 0;
            for (int j = 0; j < 20; j++) {
                total += probabilities.get(j,i);
            }
            if (total != 0) {
                for (int j = 0; j < 20; j++) {
                    probabilities.set(j, i, (probabilities.get(j, i) / total));
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