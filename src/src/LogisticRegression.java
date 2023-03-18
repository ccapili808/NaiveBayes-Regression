import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Matrix;
import no.uib.cipr.matrix.MatrixEntry;
import no.uib.cipr.matrix.io.MatrixVectorReader;
import no.uib.cipr.matrix.sparse.LinkedSparseMatrix;
import org.ojalgo.array.Array2D;
import org.ojalgo.data.DataProcessors;
import org.ojalgo.function.PrimitiveFunction;
import org.ojalgo.function.UnaryFunction;
import org.ojalgo.function.VoidFunction;
import org.ojalgo.function.aggregator.Aggregator;
import org.ojalgo.function.constant.PrimitiveMath;
import org.ojalgo.matrix.store.SparseStore;
import org.ojalgo.random.Normal;
import org.ojalgo.structure.Access1D;
import org.ojalgo.structure.Factory1D;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;
import java.util.function.Consumer;

import static no.uib.cipr.matrix.Matrices.getArray;
import static org.ojalgo.function.constant.PrimitiveMath.*;


public class LogisticRegression {
    private String vocabularyFile = "vocabulary.txt";
    private String trainingFile = "training.csv";
    private String classificationFile = "newsgrouplabels.txt";
    private String testingFile = "testing.csv";

    private LinkedSparseMatrix xMatrix = new LinkedSparseMatrix(10000,61189);
    private DenseMatrix classificationsMatrix = new DenseMatrix(12000,1);
    private DenseMatrix weightsMatrix = new DenseMatrix(20,61189);
    private DenseMatrix probabilities = new DenseMatrix(20,10000);
    private DenseMatrix deltaMatrix = new DenseMatrix(20,10000);
    private DenseMatrix lineVector = new DenseMatrix(1,61189);
    private LinkedSparseMatrix testMatrix = new LinkedSparseMatrix(2000,61189);



    //the learning rate
    private float eta;

    //the penalty term
    private float lambda;

    //the number of iterations
    private final int ITERATIONS = 10000;
    private double testAccuracy = 0;


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
                weightsMatrix.set(i,j,((0.1*r.nextFloat())));
            }
        }
        System.out.println("Reading training set...");
        Scanner sc = null;
        sc = new Scanner(new File(trainingFile));
        //read every line in training set and build the data needed
        //while (sc.hasNextLine()) {

        for(int l = 0; l<10000;l++) {
            String[] line = sc.nextLine().split(",");
            float classification = Float.parseFloat(line[line.length-1]);
            int documentID = Integer.parseInt(line[0]);
            //print out the document number to see where the code is at
            if(documentID%100 == 0) {
                System.out.println("Reading DocumentID: " + documentID);
            }
            for (int j = 1; j<21; j++) {
                //set the delta matrix
                if (classification == j) {
                    deltaMatrix.set(j-1,documentID-1,1f);
                }
                else {
                    deltaMatrix.set(j-1,documentID-1,0.0f);
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
            //This code would standardize per row of data
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

        //NormalizeMatrix();
        //StandardizeMatrix();
        //TfIdfScaling();

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
        /*
        double[] test = new double[20];
        for(int i = 10000; i<12000;i++) {
            double classification = classificationsMatrix.get(i,0);
            test[(int)(classification-1)] = test[(int)(classification-1)] + 1;
        }
         */
    }

    private void StandardizeMatrix() {
        DenseMatrix columnHelper = new DenseMatrix(10000,1);
        Consumer<MatrixEntry> set1 = a -> a.set(1);
        columnHelper.forEach(set1);
        DenseMatrix columnMeans = new DenseMatrix(61189,1);
        DenseMatrix columnSD = new DenseMatrix(61189,1);
        for (int p=0;p<61189;p++) {
            double standardDeviation = 0;
            for(int q=0;q<10000;q++) {
                standardDeviation += Math.pow(xMatrix.get(q,p)-columnMeans.get(p,0),2);
            }
            columnSD.set(p,0,Math.sqrt(standardDeviation/10000));
        }
        xMatrix.transAmult((1/10000),columnHelper,columnMeans);
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

    private void TfIdfScaling() {
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

    public void train(){
        //iterate and update weight matrix every time
        for(int k = 0; k< ITERATIONS; k++){
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
            array2D.modifyAny(DataProcessors.SCALE);
            weightsMatrix = new DenseMatrix(array2D.toRawCopy2D());
            checkAccuracy();
        }
    }
    public void checkAccuracy(){
        //multiply weights by x transpose
        //temp = weightsMatrix.multiply(xMatrix.transpose());
        DenseMatrix temp = new DenseMatrix(2000,20);
        probabilities = new DenseMatrix(20,2000);

        //make every element e^i
        testMatrix.transBmult(weightsMatrix,temp);
        temp.transpose(probabilities);
        //for(int i = 0; i < 4000; i++) {
        //    probabilities.set(19,i,0);
        //}
        Array2D<Double> array2D = Array2D.R064.rows(getArray(probabilities));
        array2D.modifyAny(DataProcessors.SCALE);
        array2D.modifyAll(EXP);
        probabilities = new DenseMatrix(array2D.toRawCopy2D());

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
            if(prediction==classificationsMatrix.get(i+10000,0)) {
                accuracy++;
            }
        }
        testAccuracy = accuracy/2000;
        System.out.println("Test accuracy: "+ testAccuracy);
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
                if (wordCount>0) {
                    xMatrix.set(documentID - 12001, i, wordCount);
                }
            }
        }

        DenseMatrix temp = new DenseMatrix(6774,20);

        xMatrix.transBmult(weightsMatrix,temp);
        temp.transpose(probabilities);


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

        DenseMatrix temp = new DenseMatrix(10000,20);
        probabilities = new DenseMatrix(20,10000);

        xMatrix.transBmult(weightsMatrix,temp);
        temp.transpose(probabilities);

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
        System.out.println("Train accuracy: "+ testAccuracy);


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