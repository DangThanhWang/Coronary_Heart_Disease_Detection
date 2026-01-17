import java.lang.Exception;
import javax.swing.UIManager;
import java.util.Random;
import java.time.Instant;
import java.io.File;

public class EcgGen extends javax.swing.JFrame {
    
    /* Main Calculation Objects */
    EcgParam paramOb;
    
    
    /* Main GUI-Window Objects*/
    EcgParamWindow paramWin;
    EcgLogWindow logWin;
    EcgPlotWindow plotWin;
    EcgCalc calcOb;
    EcgExportWindow exportWin;

    private static class GenerationScenario {
        final String folderName;
        final String activityType;
        final int label;
        final int samples;
        final double noiseMin;
        final double noiseMax;

        GenerationScenario(String folderName, String activityType, int label, int samples, double noiseMin, double noiseMax) {
            this.folderName = folderName;
            this.activityType = activityType;
            this.label = label;
            this.samples = samples;
            this.noiseMin = noiseMin;
            this.noiseMax = noiseMax;
        }
    }

    /** Creates new form ecgApplication */
    public EcgGen() {
        initClasses();
    }

    /*
     * Init Child Classes
     */
    private void initClasses(){
        // init parameter
        paramOb = new EcgParam();
        logWin = new EcgLogWindow();
        calcOb = new EcgCalc(paramOb, logWin);
        exportWin = new EcgExportWindow(null, true, paramOb, calcOb, logWin);
    }   
    
    public EcgPlotWindow getPlotWin() {
        return plotWin;
    }

    public EcgCalc getCalcOb() {
        return calcOb;
    }

    public EcgParam getParamOb() {
        return paramOb;
    }

    public EcgLogWindow getLog() {
        return logWin;
    }

    public EcgExportWindow getExportWindow() {
        return exportWin;
    }
    
    private static double randomInRange(Random random, double min, double max) {
        return min + (max - min) * random.nextDouble();
    }

        /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        
        EcgGen EcgGen = new EcgGen();
        EcgParam paramController = EcgGen.getParamOb();
        EcgLogWindow logger = EcgGen.getLog();
        EcgExportWindow dataExporter = EcgGen.getExportWindow(); 
        EcgCalc generator = EcgGen.getCalcOb();

        String baseOutputRoot = "../Data/Generated";

        GenerationScenario[] scenarios = new GenerationScenario[] {
            new GenerationScenario("Env1", "Resting-Normal", 0, 200, 0.06, 0.12),
            new GenerationScenario("Env2", "Resting-Abnormal", 1, 200, 0.07, 0.16),
            new GenerationScenario("Env3", "Working-Normal", 2, 200, 0.08, 0.18),
            new GenerationScenario("Env4", "Working-Abnormal", 3, 200, 0.10, 0.22),
            new GenerationScenario("Eval", "Resting-Normal", 0, 80, 0.05, 0.14),
            new GenerationScenario("Eval", "Resting-Abnormal", 1, 80, 0.06, 0.16),
            new GenerationScenario("Eval", "Working-Normal", 2, 80, 0.08, 0.20),
            new GenerationScenario("Eval", "Working-Abnormal", 3, 80, 0.10, 0.24),
            new GenerationScenario("Eval", "Working-Overlap", 3, 80, 0.12, 0.26)
        };

        Random random = new Random();
        Instant timestamp;

        for (GenerationScenario scenario : scenarios) {
            File logDirectory = new File(baseOutputRoot + "/" + scenario.folderName + "/logs");
            File csvDirectory = new File(baseOutputRoot + "/" + scenario.folderName + "/csv");
            logDirectory.mkdirs();
            csvDirectory.mkdirs();

            System.out.println("Generating " + scenario.samples + " samples for " + scenario.folderName + " (" + scenario.activityType + ")");

            for (int i = 0; i < scenario.samples; i++) {
                timestamp = Instant.now();

                paramController.resetParameters();
                paramController.setRandomHrStd(scenario.activityType, random);
                paramController.setRandomHrMean(scenario.activityType, random);
                paramController.setRandomLfHfRatio(scenario.activityType, random);
                paramController.setRandomSeed(random);
                paramController.setANoise(randomInRange(random, scenario.noiseMin, scenario.noiseMax));
                paramController.setRandomFLo(scenario.activityType, random);
                paramController.setRandomFHi(scenario.activityType, random);
                paramController.setRandomAForR(random);
                paramController.setRandomBForR(random);
                
                if (paramController.checkParameters()) {
                    Boolean genSuccess = generator.calculateEcg();
                    if (genSuccess) {
                        String timestampString = timestamp.toString().replace("/", "").replace(":", "_").replace(".", "_");
                        String sanitizedActivity = scenario.activityType.replace(" ", "-");
                        String desFilename = String.format("%s_%d_%s_%04d", timestampString, scenario.label, sanitizedActivity, i);

                        File logFile = new File(logDirectory, desFilename + ".txt");
                        File csvFile = new File(csvDirectory, desFilename + ".csv");

                        logger.exportTxtLog(logFile.getPath());
                        dataExporter.exportCsvData(csvFile.getPath());
                    }
                }
            }
        }
    }
}
