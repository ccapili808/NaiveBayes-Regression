import java.io.*;
import java.util.*;

public class Main {

    public static void main(String[] args){
        try {
            NaiveBayes naiveBayes = new NaiveBayes("default");
        } catch (UnsupportedEncodingException e) {
            throw new RuntimeException(e);
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
    }

}
