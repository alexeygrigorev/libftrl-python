package ololo.ftrl;

import ololo.ftrl.ololo.Metrics;
import ololo.ftrl.ololo.data.Dataset;
import ololo.ftrl.ololo.data.Line;

import java.util.ArrayList;
import java.util.List;

public class Test {
    public static void main(String[] args) throws Exception {
        FtrlProximalModel model = new FtrlProximalModel(1, 1, 0, 0, 100000);

        String path = "/home/alexey/tmp/criteo/criteo_train_small.txt.gz";

        int cnt = 0;
        int total = 0;


        try (Dataset lines = new Dataset(path, true)) {
            while (lines.hasNext()) {
                Line next = lines.next();
                int[] features = next.getFeatures();
                byte click = next.getClick();
                model.fit(features, click);
            }
        }

        List<Float> actual = new ArrayList<>();
        List<Float> predicted = new ArrayList<>();

        try (Dataset lines = new Dataset(path, true)) {
            while (lines.hasNext    ()) {
                Line next = lines.next();
                int[] features = next.getFeatures();
                byte click = next.getClick();
                actual.add((float) click);

                float pred = model.predict(features);
                predicted.add(pred);
            }
        }

        double auc = Metrics.auc(actual, predicted);
        System.out.println(auc);

    }
}
