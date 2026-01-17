/*
 * EcgParam.java
 *
 * See EcgLicense.txt for License terms.
 */

/**
 *
 * @author  Mauricio Villarroel (m.villarroel@acm.og)
 * Part of of this code was taken from the ECG C version 
 * @Copyright Patrick McSharry and Gari Clifford.
 */
import java.util.Random;

public class EcgParam {
    
    /****************************************************************
     *  GLOBAL ECG PARAMETERS:                                                  
     ****************************************************************/
    private int    N;               /*  Number of heart beats              */
    private double hrstd;           /*  Heart rate std                     */
    private double hrmean;          /*  Heart rate mean                    */
    private double lfhfratio;       /*  LF/HF ratio                        */
    private int    sfecg;           /*  ECG sampling frequency             */        
    private int    sf;              /*  Internal sampling frequency        */
    private double amplitude;       /*  Amplitude for the plot area        */
    private int seed;               /*  Seed                               */    
    private double anoise;          /*  Amplitude of additive uniform noise*/
    private int    period;
    /* Define frequency parameters for rr process 
     * flo and fhi correspond to the Mayer waves and respiratory rate respectively
     */
    private double flo;             /*  Low frequency                      */
    private double fhi;             /*  High frequency == RESPIRATORY FREQUENCY */
    private double flostd;          /*  Low frequency std                  */
    private double fhistd;          /*  High frequency std                 */
    /* Order of extrema: [P Q R S T]  */
    private double[] theta = new double[5]; /* ti not in radians*/
    private double[] a = new double[5];
    private double[] b = new double[5];

    /* Animation variables */
    private long ecgAnimateInterval;

    /* Flag to know if all parameters are valid */
    private boolean allParametersValid;

    /** Creates a new instance of EcgParam */
    public EcgParam() {
        resetParameters();
    }

    /****************************************************************
     *  GLOBAL Set/Get Parameter Functions:
     ****************************************************************/
    public void setN(int value){
        N = value;
        allParametersValid = false;
    }
    public int getN(){
        return N;
    }
    
    public void setHrStd(double value){
        hrstd = value;
        allParametersValid = false;
    }
    public double getHrStd(){
        return hrstd;
    }
    public void setRandomHrStd(String type, Random random){
        double value = 0;
        double defaultNum;
        double range;
        double min;
        double max;
        
        switch(type) {
            case "Resting-Normal":
                defaultNum = 1.6475;
                range = 0.1;
                min = defaultNum - range;
                max = defaultNum + range;
                
                value = min + (max - min) * random.nextDouble();
                break;
            case "Resting-Abnormal":
                defaultNum = 1.3693;
                range = 0.1;
                min = defaultNum - range;
                max = defaultNum + range;
                
                value = min + (max - min) * random.nextDouble();
                break;
            case "Resting-Overlap":
                defaultNum = 1.5084;
                range = 0.04;
                min = defaultNum - range;
                max = defaultNum + range;
                
                value = min + (max - min) * random.nextDouble();
                break;
            case "Working-Normal":
                defaultNum = 2.211;
                range = 0.1;
                min = defaultNum - range;
                max = defaultNum + range;
                
                value = min + (max - min) * random.nextDouble();
                break;
            case "Working-Abnormal":
                defaultNum = 2.494;
                range = 0.1;
                min = defaultNum - range;
                max = defaultNum + range;
                
                value = min + (max - min) * random.nextDouble();
                break;
            case "Working-Overlap":
                defaultNum = 2.3525;
                range = 0.04;
                min = defaultNum - range;
                max = defaultNum + range;
                
                value = min + (max - min) * random.nextDouble();
                break;
            default:
                System.out.print("Wrong type");
                // code block
          }
          
        hrstd = value;
        allParametersValid = false;
    }

    

    public void setHrMean(double value){
        hrmean = value;
        allParametersValid = false;
    }
    public double getHrMean(){
        return hrmean;
    }
    public void setRandomHrMean(String type, Random random){
        double value = 0;
        double min;
        double max;
        
        switch(type) {
            case "Resting-Normal":
                min = 50.00;
                max = 56.00;
                
                value = min + (max - min) * random.nextDouble();
                break;
            case "Resting-Abnormal":
                min = 60.00;
                max = 65.00;
                
                value = min + (max - min) * random.nextDouble();
                break;
            case "Resting-Overlap":
                min = 56.50;
                max = 59.50;
                
                value = min + (max - min) * random.nextDouble();
                break;
            case "Working-Normal":
                min = 65.00;
                max = 73.00;
                
                value = min + (max - min) * random.nextDouble();
                break;
            case "Working-Abnormal":
                min = 126.00;
                max = 140.00;
                
                value = min + (max - min) * random.nextDouble();
                break;
            case "Working-Overlap":
                min = 73.00;
                max = 126.00;
                
                value = min + (max - min) * random.nextDouble();
                break;
            case "Exercising":
                break;
            default:
                System.out.print("Wrong type");
                // code block
          }
          
        hrmean = value;
        allParametersValid = false;
    }


    public void setLfHfRatio(double value){
        lfhfratio = value;
        allParametersValid = false;
    }
    public double getLfHfRatio(){
        return lfhfratio;
    }
    public void setRandomLfHfRatio(String type, Random random){
        double value = 0;
        double defaultNum;
        double range;
        double min;
        double max;

        
        switch(type) {
            case "Resting-Normal":
                defaultNum = 0.5;
                range = 0.15;
                min = defaultNum - range;
                max = defaultNum + range;
                
                value = min + (max - min) * random.nextDouble();
                break;
            case "Resting-Abnormal":
                defaultNum = 0.5;
                range = 0.15;
                min = defaultNum - range;
                max = defaultNum + range;
                
                value = min + (max - min) * random.nextDouble();
                break;
            case "Resting-Overlap":
                defaultNum = 0.5;
                range = 0.15;
                min = defaultNum - range;
                max = defaultNum + range;
                
                value = min + (max - min) * random.nextDouble();
                break;
            case "Working-Normal":
                defaultNum = 1;
                range = 0.15;
                min = defaultNum - range;
                max = defaultNum + range;
                
                value = min + (max - min) * random.nextDouble();
                break;
            case "Working-Abnormal":
                defaultNum = 1;
                range = 0.15;
                min = defaultNum - range;
                max = defaultNum + range;
                
                value = min + (max - min) * random.nextDouble();
                break;
            case "Working-Overlap":
                defaultNum = 1;
                range = 0.15;
                min = defaultNum - range;
                max = defaultNum + range;
                
                value = min + (max - min) * random.nextDouble();
                break;
            case "Exercising":
                break;
            default:
                System.out.print("Wrong type");
          }
          
        lfhfratio = value;
        allParametersValid = false;
    }


    public void setSfEcg(int value){
        sfecg = value;
        ecgAnimateInterval = (long)(1000/(sfecg));
        allParametersValid = false;
    }
    public int getSfEcg(){
        return sfecg;
    }

    public void setSf(int value){
        sf = value;
        allParametersValid = false;
    }
    public int getSf(){
        return sf;
    }

    public void setAmplitude(double value){
        amplitude = value;
        allParametersValid = false;
    }
    public double getAmplitude(){
        return amplitude;
    }

    public void setSeed(int value){
        seed = value;
        allParametersValid = false;
    }
    public int getSeed(){
        return seed;
    }
    public void setRandomSeed(Random random){                
        // Generate a random integer within a specified range
        int min = 1;
        int max = 42;
        seed = random.nextInt(max - min + 1) + min; // Random int between 1 and 42
          
        allParametersValid = false;
    }


    public void setANoise(double value){
        anoise = value;
        allParametersValid = false;
    }
    public double getANoise(){
        return anoise;
    }
    public void setRandomANoise(Random random){        
        double defaultNum = 0.1;
        double range = 0.01;
        double min = defaultNum - range;
        double max = defaultNum + range;
        
        double value = min + (max - min) * random.nextDouble();
    
        anoise = value;
        allParametersValid = false;
    }


    public void setPeriod(int value){
        period = value;
        allParametersValid = false;
    }
    public int getPeriod(){
        return period;
    }

    public void setFLo(double value){
        flo = value;
        allParametersValid = false;
    }
    public double getFLo(){
        return flo;
    }
    public void setRandomFLo(String type, Random random){
        double value = 0;
        double defaultNum;
        double range;
        double min;
        double max;

        switch(type) {
            case "Resting-Normal":
                defaultNum = 0.1;
                range = 0.03;
                min = defaultNum - range;
                max = defaultNum + range;
                
                value = min + (max - min) * random.nextDouble();
                break;
            case "Resting-Abnormal":
                defaultNum = 0.1;
                range = 0.03;
                min = defaultNum - range;
                max = defaultNum + range;
                
                value = min + (max - min) * random.nextDouble();
                break;
            case "Resting-Overlap":
                defaultNum = 0.1;
                range = 0.03;
                min = defaultNum - range;
                max = defaultNum + range;
                
                value = min + (max - min) * random.nextDouble();
                break;
            case "Working-Normal":
                defaultNum = 0.25;
                range = 0.03;
                min = defaultNum - range;
                max = defaultNum + range;
                
                value = min + (max - min) * random.nextDouble();
                break;
            case "Working-Abnormal":
                defaultNum = 0.25;
                range = 0.03;
                min = defaultNum - range;
                max = defaultNum + range;
                
                value = min + (max - min) * random.nextDouble();
                break;
            case "Working-Overlap":
                defaultNum = 0.25;
                range = 0.03;
                min = defaultNum - range;
                max = defaultNum + range;
                
                value = min + (max - min) * random.nextDouble();
                break;

            case "Working":
                break;
            case "Exercising":
                break;
            default:
                System.out.print("Wrong type");
                // code block
          }

    
        flo = value;
        allParametersValid = false;
    }


    public void setFHi(double value){
        fhi = value;
        allParametersValid = false;
    }
    public double getFHi(){
        return fhi;
    }
    public void setRandomFHi(String type, Random random){
        double value = 0;
        double defaultNum;
        double range;
        double min;
        double max;

        switch(type) {
            case "Resting-Normal":
                defaultNum = 0.25;
                range = 0.05;
                min = defaultNum - range;
                max = defaultNum + range;
                
                value = min + (max - min) * random.nextDouble();
                break;
            case "Resting-Abnormal":
                defaultNum = 0.25;
                range = 0.05;
                min = defaultNum - range;
                max = defaultNum + range;
                
                value = min + (max - min) * random.nextDouble();
                break;
            case "Resting-Overlap":
                defaultNum = 0.25;
                range = 0.05;
                min = defaultNum - range;
                max = defaultNum + range;
                
                value = min + (max - min) * random.nextDouble();
                break;
            case "Working-Normal":
                defaultNum = 0.25;
                range = 0.03;
                min = defaultNum - range;
                max = defaultNum + range;
                
                value = min + (max - min) * random.nextDouble();
                break;
            case "Working-Abnormal":
                defaultNum = 0.25;
                range = 0.03;
                min = defaultNum - range;
                max = defaultNum + range;
                
                value = min + (max - min) * random.nextDouble();
                break;
            case "Working-Overlap":
                defaultNum = 0.25;
                range = 0.03;
                min = defaultNum - range;
                max = defaultNum + range;
                
                value = min + (max - min) * random.nextDouble();
                break;

            case "Working":
                break;
            case "Exercising":
                break;
            default:
                System.out.print("Wrong type");
                // code block
          }

    
        fhi = value;
        allParametersValid = false;
    }


    public void setFLoStd(double value){
        flostd = value;
        allParametersValid = false;
    }
    public double getFLoStd(){
        return flostd;
    }

    public void setFHiStd(double value){
        fhistd = value;
        allParametersValid = false;
    }
    public double getFHiStd(){
        return fhistd;
    }

    public void setTheta(int index, double value){
        theta[index] = value;
        allParametersValid = false;
    }
    public double getTheta(int index){
        return theta[index];
    }

    public void setA(int index, double value){
        a[index] = value;
        allParametersValid = false;
    }
    public double getA(int index){
        return a[index];
    }
    public void setRandomAForR(Random random) {                
        // Generate a random integer within a specified range
        int min = 27;
        int max = 44;
        seed = random.nextInt(max - min + 1) + min; // Random int between 1 and 42
        
        setA(2, seed);
    }


    public void setB(int index, double value){
        b[index] = value;
        allParametersValid = false;
    }
    public double getB(int index){
        return b[index];
    }
    public void setRandomBForR(Random random) {                
        // Generate a random integer within a specified range
        double min = 0.1 - 0.02;
        double max = 0.1 + 0.02;
        double value = min + (max - min) * random.nextDouble();
        
        setA(2, value);
    }


    public void setEcgAnimateInterval(long value){
        ecgAnimateInterval = value;
        allParametersValid = false;
    }
    public long getEcgAnimateInterval(){
        return ecgAnimateInterval;
    }
    /****************************************************************
     *  Functions:
     ****************************************************************/

    /*
     * check to see if all parameters are valid
     */
    public boolean isValid(){
        return allParametersValid;
    }

    /*
     * In this function, it can be enforced
     * adicional rules.
     */
    public boolean checkParameters(){

        boolean RetValue = true;
        allParametersValid = true;

        // Check the Internal frequency respect to ECG frequency
        if(((int)Math.IEEEremainder(sf, sfecg)) != 0){
                RetValue = false;
                allParametersValid = false;
        }

        return RetValue;
    }
    /*
     * ReInit the Button Parameters' values
     */
    public void resetParameters(){
        /* General Intergace parameters */
        N = 256;            //number of heart rates
        sfecg = 256;        //ECG sampling frequency    
        sf = 512;           //internal sampling frequency
        anoise = 0.1;      //Amplitude of additive uniform noise
        hrmean = 60.0;      //Heart rate mean
        hrstd = 1.0;        //Heart rate standard deviation
        seed = 1;           
        amplitude = 1.4;    //plot area amplitude
        flo = 0.1;          //low frequency
        fhi = 0.25;         //high frequency
        flostd = 0.01;      //low frequency standard deviation
        fhistd = 0.01;      //high frequency standard deviation
        lfhfratio = 0.5;    //LF/HF ratio

        /* ECG morphology: Order of extrema: [P Q R S T] */
        theta[0]= -60.0;    
        theta[1]= -15.0;
        theta[2]= 0.0;
        theta[3]= 15.0;
        theta[4]= 90.0;

        a[0]= 1.2;
        a[1]= -5.0;
        a[2]= 30.0;
        a[3]= -7.5;
        a[4]= 0.75;
        
        b[0]= 0.25;
        b[1]= 0.1;
        b[2]= 0.1;
        b[3]= 0.1;
        b[4]= 0.4;
        
        ecgAnimateInterval = (long)(1000/(sfecg));

        allParametersValid = true;
    }
}
