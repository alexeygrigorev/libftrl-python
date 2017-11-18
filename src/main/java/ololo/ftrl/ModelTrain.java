package ololo.ftrl;

import com.google.common.base.Stopwatch;
import ololo.ftrl.ololo.data.BinaryDataset;
import ololo.ftrl.ololo.data.Line;
import org.apache.commons.io.output.NullOutputStream;

import java.io.PrintWriter;

public class ModelTrain {
    public static void main(String[] args) throws Exception {
        float alpha = 1.0f;
        float beta = 1.0f;
        float l1 = 20.0f;
        float l2 = 0.0f;

        int numPasses = 6;

        boolean validation = false;

        PrintWriter logger = new PrintWriter(new NullOutputStream());
        if (validation) {
            String logName = String.format("loss_a=%.3f_b=%.3f_l1=%.3f_l2=%.3f.txt", alpha, beta, l1, l2);
            logger = new PrintWriter(logName);
        }

        FtrlProximalModel model = new FtrlProximalModel(alpha, beta, l1, l2, 100000);

        String trainPath = "/home/alexey/tmp/criteo/criteo_train_clicks.bin";

        int winSize = 100000;
        float[] window = new float[winSize];
        int wCnt = 0;

        int i = 0;

        for (int t = 0; t < numPasses; t++) {
            Stopwatch pass = Stopwatch.createStarted();
            try (BinaryDataset lines = new BinaryDataset(trainPath)) {
                while (lines.hasNext()) {
                    Line next = lines.next();
                    int[] features = next.getFeatures();
                    byte click = next.getClick();

                    i++;

                    if (!validation) {
                        model.fit(features, click);

                        if (i % 100000 == 0) {
                            System.out.println(i);
                        }

                        continue;
                    }

                    if (next.getGroupId() % 10 == 0) {
                        model.fit(features, click);
                    } else {
                        float predict = model.predict(features);
                        float loss = FtrlProximalModel.logloss(predict, click);
                        window[wCnt] = loss;
                        wCnt = (wCnt + 1) % winSize;
                    }

                    if (i > 100000 && i % 100000 == 0) {
                        System.out.println(i + " " + avg(window));
                        logger.println(i + " " + avg(window));
                    }
                }
            }

            System.out.printf("pass finished in %s%n", pass.stop());
        }

        logger.flush();
        logger.close();

        String modelFile = String.format("model_n=%d_a=%.3f_b=%.3f_l1=%.3f_l2=%.3f.bin", numPasses, alpha, beta, l1, l2);
        model.save(modelFile);
    }

    private static float avg(float[] window) {
        float sum = 0.0f;
        for (int i = 0; i < window.length; i++) {
            sum = sum + window[i];
        }
        return sum / window.length;
    }
}
