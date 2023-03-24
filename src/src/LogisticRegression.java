import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.MatrixEntry;
import no.uib.cipr.matrix.sparse.LinkedSparseMatrix;
import org.ojalgo.array.Array2D;
import org.ojalgo.data.DataProcessors;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;
import java.util.function.Consumer;

import static no.uib.cipr.matrix.Matrices.getArray;
import static org.ojalgo.function.constant.PrimitiveMath.*;

/**
 * This class trains a Logistic Regression model for the
 * 20newsgroups data set. It uses matrix linear algebra operations
 * from libraries ojAlgo (https://search.maven.org/artifact/org.ojalgo/ojalgo/52.0.1/jar?eh=)
 * and matrix-toolkits-java (https://search.maven.org/artifact/com.googlecode.matrix-toolkits-java/mtj/1.0.4/jar?eh=)
 * Every iteration, the iteration number, the training data accuracy, testing data accuracy
 * and conditional data likelihood are printed in the console.
 * This class uses matrices instead of standard Java structures because they are more efficient
 * and this code can perform ~3 iterations of gradient descent per second or ~9000 iterations per hour.
 */
public class LogisticRegression {
    //set the file names
    private String vocabularyFile = "vocabulary.txt";
    private String trainingFile = "training.csv";
    private String classificationFile = "newsgrouplabels.txt";
    private String testingFile = "testing.csv";
    /*
    create the matrices necessary
    the training and validation matrices are using matrix-toolkits-java's sparse
    matrix implementation to improve efficiency. The rest of the matrices are
    dense since there are not many 0 values in them. Using sparse matrices
    also improved memory usage by a lot.
     */
    private LinkedSparseMatrix xMatrix = new LinkedSparseMatrix(10000,61189);
    private DenseMatrix classificationsMatrix = new DenseMatrix(12000,1);
    private DenseMatrix weightsMatrix = new DenseMatrix(20,61189);
    private DenseMatrix probabilities = new DenseMatrix(20,10000);
    private DenseMatrix deltaMatrix = new DenseMatrix(20,10000);
    private DenseMatrix lineVector = new DenseMatrix(1,61189);
    private DenseMatrix columnMax = new DenseMatrix(61189, 1);
    private DenseMatrix columnMin = new DenseMatrix(61189, 1);
    private LinkedSparseMatrix testMatrix = new LinkedSparseMatrix(2000,61189);
    //confusion matrix array
    private int[][] confusionMatrix = new int [20][20];
    //boolean to control when the confusion matrix is calculated and printed (at the end of training)
    private boolean getConfusionMatrix = false;




    //the learning rate
    private float eta;

    //the penalty term
    private float lambda;

    //the number of iterations
    private int iterations = 1000;
    private double testAccuracy = 0;


    /**
     * Constructor for LogisticRegression
     * @param lambda the penalty value to use
     * @param eta the learning rate to use
     */
    public LogisticRegression(float lambda, float eta, int iterations){
        this.eta = eta;
        this.lambda = lambda;
        this.iterations = iterations;
        try {
            createDataSet();
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
        train();
        predict();
    }

    /**
     * This method reads in the dataset files.
     * The training file is split to 10000 examples for the training set
     * and 2000 examples for the validation set.
     * These are not randomized as it seemed like the examples are already
     * pretty randomized in the file, and I wanted to get consistent training results
     * for testing and debugging.
     * This method also calls the training matrix scaling methods.
     * @throws FileNotFoundException
     */
    public void createDataSet() throws FileNotFoundException {
        Random r = new Random();
        //initialize random weights between 0-0.1
        for(int i =0; i <20;i++) {
            for(int j = 0; j<61189;j++) {
                weightsMatrix.set(i,j,((0.1*r.nextFloat())));
            }
        }
        System.out.println("Reading training set...");
        Scanner sc = null;
        sc = new Scanner(new File(trainingFile));
        //read every line in training set and build the data needed
        for(int l = 0; l<10000;l++) {
            String[] line = sc.nextLine().split(",");
            //get the classification
            float classification = Float.parseFloat(line[line.length-1]);
            //get the document ID
            int documentID = Integer.parseInt(line[0]);
            //print out the document number to see where the code is at
            if(documentID%100 == 0) {
                System.out.println("Reading DocumentID: " + documentID);
            }
            //set the delta matrix per Mitchell book function
            for (int j = 1; j<21; j++) {
                //set the delta matrix
                if (classification == j) {
                    deltaMatrix.set(j-1,documentID-1,1f);
                }
                else {
                    deltaMatrix.set(j-1,documentID-1,0.0f);
                }
            }
            //add the document's classification to the classifications matrix
            classificationsMatrix.set(documentID-1,0,classification);
            //set x0 to 1 for the bias
            xMatrix.set(documentID-1,0,1f);
            //get column mins and column maxes for MinMaxNormalization
            Consumer<MatrixEntry> add300 = a -> a.set(a.get()+300);
            columnMin.forEach(add300);
            columnMin.set(0,0,1);
            columnMax.set(0,0,1);
            for (int i=1; i < (line.length-2); i++) {
                float wordCount = Float.parseFloat(line[i]);
                //add specific word count from each document to the total count of that word to the example matrix
                if(wordCount>0) {
                    xMatrix.set(documentID-1,i,wordCount);
                }
                //update column max if necessary
                if (wordCount>columnMax.get(i,0)) {
                    columnMax.set(i,0,wordCount);
                }
                //update column min if necessary
                if (wordCount<columnMin.get(i,0)) {
                    columnMin.set(i,0,wordCount);
                }
            }
            //This code would standardize per row of data but was mostly used for testing
            //I think scaling per column is better for this problem
            /*

            lineVector.zero();
            for (int i=1; i < (line.length-2); i++) {
                float wordCount = Float.parseFloat(line[i]);
                //add specific word count from each document to the total count of that word to the example matrix
                if(wordCount>0) {
                    lineVector.set(0,i,wordCount);
                }
                    //xMatrix.set(documentID-1,i,wordCount);
            }
            double norm = lineVector.norm(Matrix.Norm.Frobenius);
            Consumer<MatrixEntry> entry = a -> a.set((a.get()/norm));
            //Consumer<MatrixEntry> entry = a -> a.set(Math.log(a.get()+1));
            lineVector.forEach(entry);
            for(int k=1;k<61189;k++) {
                double wordCount = lineVector.get(0,k);
                if (wordCount>0) {
                    xMatrix.set(documentID-1,k,wordCount);
                }
            }
            */
        }
        /*
        These are the different scaling methods I implemented for the X matrix.
        We found that TfIdf Scaling followed by my normalization method was the best
        for accuracy. This will be discussed more in the report.
        */
        TfIdfScaling();
        NormalizeMatrix();
        /*
        The other two scaling methods are available to use but are not used for our final model
        so they are commented out.
         */
        //StandardizeMatrix();
        //MinMaxNormalize();

        //build validation set
        for(int l = 10000; l<12000;l++) {
            String[] line = sc.nextLine().split(",");
            int classification = Integer.parseInt(line[line.length-1]);
            int documentID = Integer.parseInt(line[0]);
            //print out the document number to see where the code is at
            if(documentID%100 == 0) {
                System.out.println("Reading DocumentID: " + documentID);
            }
            //add the document's classification to the matrix
            classificationsMatrix.set(documentID-1,0,classification);
            //set x0 to 1
            testMatrix.set(documentID-10001,0,1);
            for (int i=1; i < (line.length-2); i++) {
                double wordCount = Double.parseDouble(line[i]);
                //add specific word count from each document to the total count of that word to the example matrix
                if(wordCount>0) {
                    testMatrix.set(documentID-10001,i,wordCount);
                }
            }
        }
    }

    /**
     * This method gets the standard score for each value of the x matrix column wise.
     */
    private void StandardizeMatrix() {
        DenseMatrix columnHelper = new DenseMatrix(10000,1);
        Consumer<MatrixEntry> set1 = a -> a.set(1);
        columnHelper.forEach(set1);
        DenseMatrix columnMeans = new DenseMatrix(61189,1);
        DenseMatrix columnSD = new DenseMatrix(61189,1);
        //calculate column means by summing each column of X and dividing by 10000
        xMatrix.transAmult((1/10000),columnHelper,columnMeans);
        //get standard deviations for each column of data
        for (int p=0;p<61189;p++) {
            double standardDeviation = 0;
            for(int q=0;q<10000;q++) {
                standardDeviation += Math.pow(xMatrix.get(q,p)-columnMeans.get(p,0),2);
            }
            columnSD.set(p,0,Math.sqrt(standardDeviation/10000));
        }
        //iterate through the x matrix set each x value to the standard score
        Iterator<MatrixEntry> test = xMatrix.iterator();
        while (test.hasNext()) {
            MatrixEntry next = test.next();
            if (next.column()==1) {
                xMatrix.set(next.row(),next.column(),1);
            }
            else {
                xMatrix.set(next.row(), next.column(), (next.get() -columnMeans.get(next.column(),0))
                        / (columnSD.get(next.column(),0)));
            }
        }
    }

    /**
     * This method scales the matrix column-wise in the following way:
     * Each value of x is divided by the total sum of each column.
     * This way, all the values of a column add up to 1.
     * I suspect this performed well because it penalizes common English words
     * that have very high values.
     */
    private void NormalizeMatrix() {
        DenseMatrix columnHelper = new DenseMatrix(10000,1);
        Consumer<MatrixEntry> set1 = a -> a.set(1);
        columnHelper.forEach(set1);
        DenseMatrix columnSums = new DenseMatrix(61189,1);
        xMatrix.transAmult(columnHelper,columnSums);
        Iterator<MatrixEntry> test = xMatrix.iterator();
        while (test.hasNext()) {
            MatrixEntry next = test.next();
            xMatrix.set(next.row(), next.column(), next.get() / columnSums.get(next.column(), 0));
        }
    }

    /**
     * This method performs MinMax normalization on the x matrix
     */
    private void MinMaxNormalize() {
        Iterator<MatrixEntry> test = xMatrix.iterator();
        while (test.hasNext()) {
            MatrixEntry next = test.next();
            if (next.column()!=0) {
                double max = columnMax.get(next.column(), 0);
                double min = columnMin.get(next.column(), 0);
                double normalizedValue;
                if (max == min) {
                    normalizedValue = 0;
                }
                else {
                    normalizedValue = (next.get() - min) / (max - min);
                }
                xMatrix.set(next.row(), next.column(), normalizedValue);
            }
        }
    }

    /**
     * This method performs TfIdf scaling on the x matrix
     */
    private void TfIdfScaling() {
        System.out.println("Performing Tf-Idf scaling on matrix");
        DenseMatrix wordSum = new DenseMatrix(10000,1);
        DenseMatrix sumHelper = new DenseMatrix(61189,1);
        Consumer<MatrixEntry> set1 = a -> a.set(1);
        sumHelper.forEach(set1);
        xMatrix.mult(sumHelper, wordSum);
        Consumer<MatrixEntry> sub1 = a -> a.set(a.get()-1);
        wordSum.forEach(sub1);
        DenseMatrix wordAppears = new DenseMatrix(61189,1);
        for(int i=1;i<61189;i++) {
            double totalDocs = 0;
            for(int j=0;j<10000;j++) {
                if (xMatrix.get(j,i)>0) {
                    totalDocs += 1;
                }
            }
            wordAppears.set(i,0,totalDocs);
        }
        Iterator<MatrixEntry> test = xMatrix.iterator();
        while (test.hasNext()) {
            MatrixEntry next = test.next();
            int column = next.column();
            int row = next.row();
            if (column!=0) {
                double tfidf = ((xMatrix.get(row,column))/wordSum.get(row,0))*Math.log((10000+1)/(wordAppears.get(column,0)+1));

                xMatrix.set(next.row(), next.column(), tfidf);
            }
        }

    }


    /**
     * This method trains the Logistic Regression model
     */
    public void train(){
        //iterate and update weight matrix every time
        for(int k = 0; k< iterations; k++){
            System.out.println("Iteration Number: " + k);
            //calculate the new probability matrix using the weights
            calculateProbabilities();

            //update weights
            DenseMatrix temp;
            DenseMatrix temp2 = new DenseMatrix(61189,20);
            DenseMatrix temp3 = new DenseMatrix(20,61189);
            //copy the delta matrix
            temp = deltaMatrix.copy();
            //subtract probabilities matrix from delta matrix
            temp.add(-1, probabilities);
            //multiply the above matrix with X using transposes and matrix rules
            //using the sparse matrix on the left side of the multiplication is necessary
            //for sparse matrix performance
            xMatrix.transABmult(temp, temp2);
            temp2.transpose(temp3);
            //subtract lambda*weightsMatrix
            temp3.add((0-lambda), weightsMatrix);
            //finally add the above to our weightsMatrix to update the weights
            weightsMatrix.add(eta, temp3);
            /*
            The following 3 lines were used to standardize the weight matrix each iteration which
            I found to be unnecessary.
             */
            //Array2D<Double> array2D = Array2D.R064.rows(getArray(weightsMatrix));
            //array2D.modifyAny(DataProcessors.STANDARD_SCORE);
            //weightsMatrix = new DenseMatrix(array2D.toRawCopy2D());

            //print the confusion matrix if its the final iteration
            if (k == (iterations -1)) {
                getConfusionMatrix = true;
            }
            //check the accuracy of the model for this iteration
            checkAccuracy();
        }
    }

    /**
     * This method checks the accuracy of the model against the validation set
     */
    public void checkAccuracy(){
        DenseMatrix temp = new DenseMatrix(2000,20);
        probabilities = new DenseMatrix(20,2000);

        //multiply weights by validation set transpose
        testMatrix.transBmult(weightsMatrix,temp);
        temp.transpose(probabilities);

        /*
        make every element e^i
        scaling of the above matrix was necessary to prevent overflow
        and Infinity/NaN values
        */
        Array2D<Double> array2D = Array2D.R064.rows(getArray(probabilities));
        array2D.modifyAny(DataProcessors.SCALE);
        array2D.modifyAll(EXP);
        probabilities = new DenseMatrix(array2D.toRawCopy2D());

        //divide each probability by the sum of the column so that they all add up to 1
        for(int i = 0;i < 2000; i++) {
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


        //calculate accuracy of predictions
        double accuracy = 0;
        for(int i = 0;i < 2000; i++) {
            double argmax = 0;
            int prediction = 0;
            for (int j = 0;j < 20; j++) {
                if (probabilities.get(j,i) > argmax) {
                    argmax = probabilities.get(j,i);
                    prediction = j+1;
                }
            }
            if (getConfusionMatrix) {
                confusionMatrix[(int)classificationsMatrix.get(i+10000,0)-1][prediction-1]++;
            }
            if(prediction==classificationsMatrix.get(i+10000,0)) {
                accuracy++;
            }
        }
        testAccuracy = accuracy/2000;
        //print out accuracy
        System.out.println("Test accuracy: "+ testAccuracy);
        //print confusion matrix if last iteration
        if (getConfusionMatrix) {
            System.out.println("Confusion Matrix:");
            for (int i = 0; i < 20; i++) {
                System.out.println(Arrays.toString(confusionMatrix[i]));
            }
        }
    }

    /**
     * This method predicts the class for the testing set from Kaggle
     */
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
                if (wordCount>0) {
                    xMatrix.set(documentID - 12001, i, wordCount);
                }
            }
        }

        DenseMatrix temp = new DenseMatrix(6774,20);
        //multiply weights by test set transpose
        xMatrix.transBmult(weightsMatrix,temp);
        temp.transpose(probabilities);

        /*
        make every element e^i
        scaling of the above matrix was necessary to prevent overflow
        and Infinity/NaN values
        */
        Array2D<Double> array2D = Array2D.R064.rows(getArray(probabilities));
        array2D.modifyAny(DataProcessors.SCALE);
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

        //get the argmax for each example and print out the prediction
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

    /**
     * This method calculates the new probability matrix using the updated
     * weights each iteration
     */
    public void calculateProbabilities() {

        DenseMatrix temp = new DenseMatrix(10000,20);
        probabilities = new DenseMatrix(20,10000);

        //multiply weights by training set transpose

        xMatrix.transBmult(weightsMatrix,temp);
        temp.transpose(probabilities);

        /*
        make every element e^i
        scaling of the above matrix was necessary to prevent overflow
        and Infinity/NaN values
        */
        Array2D<Double> array2D = Array2D.R064.rows(getArray(probabilities));
        array2D.modifyAny(DataProcessors.SCALE);
        array2D.modifyAll(EXP);

        probabilities = new DenseMatrix(array2D.toRawCopy2D());

        //normalize probability columns
        for(int i = 0;i < 10000; i++) {
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

        //calculate and print conditional data likelihood.
        //we can see that it increases every iteration until it is maximized
        double logcdl = 0;
        for (int i = 0; i < 10000; i++) {
            int classification = (int)classificationsMatrix.get(i,0);
            logcdl += Math.log(probabilities.get(classification-1, i));
        }
        System.out.println("Conditional Data Likelihood: " + logcdl);

        //calculate accuracy of the model against the training set
        double accuracy = 0;
        for(int i = 0;i < 10000; i++) {
            double argmax = 0;
            int prediction = 0;
            for (int j = 0;j < 20; j++) {
                if (probabilities.get(j,i) > argmax) {
                    argmax = probabilities.get(j,i);
                    prediction = j+1;
                }
            }
            if(prediction==classificationsMatrix.get(i,0)) {
                accuracy++;
            }
        }
        testAccuracy = accuracy/10000;
        //print out training accuracy
        System.out.println("Train accuracy: "+ testAccuracy);


    }

    /**
     * This method can change the vocabulary file name
     * @param vocabularyFile the vocabulary file name
     */
    public void setVocabularyFile(String vocabularyFile) {
        this.vocabularyFile = vocabularyFile;
    }

    /**
     * This method can change the training file name
     * @param trainingFile the training file name
     */
    public void setTrainingFile(String trainingFile) {
        this.trainingFile = trainingFile;
    }

    /**
     * This method can change the classification file name
     * @param classificationFile the classification file name
     */
    public void setClassificationFile(String classificationFile) {
        this.classificationFile = classificationFile;
    }

    /**
     * This method can change the testing file name
     * @param testingFile the testing file name
     */
    public void setTestingFile(String testingFile) {
        this.testingFile = testingFile;
    }
}