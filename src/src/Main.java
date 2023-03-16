import java.io.*;
import java.util.*;

public class Main {

    public static void main(String[] args){
        while(prompt() == 1) {
            prompt();
        }
    }

    /**
     * Prompts the user to choose which algorithm to run.
     * If the user chooses Naive Bayes, they are prompted to enter a beta value.
     * If the user chooses Logistic Regression, they are prompted to enter a lambda value.
     */
    public static int prompt() {
        //Run in a loop until the user chooses to exit
        System.out.println("This program runs Naive Bayes and Logistic Regression on the Newsgroups dataset.");
        System.out.println("Please enter the number of the algorithm you would like to run:");
        System.out.println("[1] Naive Bayes");
        System.out.println("[2] Logistic Regression");
        System.out.println("[3] Exit");
        Scanner scanner = new Scanner(System.in);
        int choice = scanner.nextInt();
        switch (choice) {
            case 1 -> {
                //Naive Bayes
                String beta;
                System.out.println("Please enter the beta value you would like to use:");
                beta = scanner.next();
                try {
                    NaiveBayes naiveBayes = new NaiveBayes(beta);
                } catch (FileNotFoundException | UnsupportedEncodingException e) {
                    throw new RuntimeException(e);
                }
                break;
            }
            case 2 -> {
                //Logistic Regression
                System.out.println("Please enter the lambda value you would like to use:");
                float lambda = scanner.nextFloat();
                LogisticRegression logisticRegression = new LogisticRegression(lambda);
                break;
            }
            case 3 -> {
                //Exit
                System.out.println("Exiting...");
                return 0;
            }
            default -> {
                System.out.println("Invalid choice. Please try again.");
                prompt();
                break;
            }
        }
        scanner.close();
        return 1;
    }

}
