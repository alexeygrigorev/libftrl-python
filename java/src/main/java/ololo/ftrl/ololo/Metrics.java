package ololo.ftrl.ololo;

import com.google.common.primitives.Floats;
import org.apache.commons.lang3.Validate;
import smile.validation.AUC;

import java.util.List;

public class Metrics {
    public static double auc(List<Float> actual, List<Float> predicted) {
        return auc(Floats.toArray(actual), Floats.toArray(predicted));
    }

    public static double auc(float[] actual, float[] predicted) {
        Validate.isTrue(actual.length == predicted.length, "the lengths don't match");

        int[] truth = float2int(actual);

        double auc = AUC.measure(truth, float2double(predicted));
        if (auc > 0.5) {
            return auc;
        } else {
            return 1 - auc;
        }
    }

    private static double[] float2double(float[] in) {
        double[] res = new double[in.length];
        for (int i = 0; i < in.length; i++) {
            res[i] = in[i];
        }
        return res;
    }

    private static int[] float2int(float[] in) {
        int[] res = new int[in.length];
        for (int i = 0; i < in.length; i++) {
            res[i] = (int) in[i];
        }
        return res;
    }
}
