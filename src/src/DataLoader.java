public class DataLoader {
    private String vocabularyFile;
    private String trainingFile;
    private String classificationFile;
    private String testingFile;

    /**
     * Default constructor
     */
    public DataLoader() {
        this.vocabularyFile = "vocabulary.txt";
        this.trainingFile = "training.txt";
        this.classificationFile = "classification.txt";
        this.testingFile = "testing.txt";
    }

    public String getVocabularyFile() {
        return vocabularyFile;
    }

    public String getTrainingFile() {
        return trainingFile;
    }

    public String getClassificationFile() {
        return classificationFile;
    }

    public String getTestingFile() {
        return testingFile;
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
}
