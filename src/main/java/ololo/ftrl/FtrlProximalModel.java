package ololo.ftrl;


/**
 * https://github.com/alexeygrigorev/outbrain-click-prediction-kaggle/blob/master/ftrl.py
 */
public class FtrlProximalModel {


    private final float alpha;
    private final float beta;
    private final float l1;
    private final float l2;
    private final int numFeatures;

    private final float[] n;
    private final float[] z;
    private final float[] w;

    private final float nIntercept;
    private final float zIntercept;
    private final float wIntercept;

    public FtrlProximalModel(float alpha, float beta, float l1, float l2, int numFeatures) {
        this.alpha = alpha;
        this.beta = beta;
        this.l1 = l1;
        this.l2 = l2;
        this.numFeatures = numFeatures;

        this.n = new float[numFeatures];
        this.z = new float[numFeatures];
        this.w = new float[numFeatures];

        this.nIntercept = 0.0f;
        this.zIntercept = 0.0f;
        this.wIntercept = 0.0f;
    }

    public void fit(int[] values, float y) {
        float p = predict(values);

        for (int i : values) {
            float sign = Math.signum(z[i]);
            if (sign * z[i] <= l1) {

            }
        }
    }

    public float predict(int[] values) {
        float wtx = wIntercept;

        for (int i : values) {
            float sign = Math.signum(z[i]);
            if (sign * z[i] > l1) {
                float f = (sign * l1 - z[i]) / ((beta + sqrt(n[i])) / alpha + l2);
                wtx = wtx + f;
            }
        }

        return sigmoid(wtx);
    }

    private static  float sigmoid(float x) {
        if (x <= -35f) {
            return 0.000000000000001f;
        } else if (x >= 35f) {
            return 0.999999999999999f;
        }

        return 1.0f / (1.0f + exp(-x));
    }

    private static float sqrt(float f) {
        return (float) Math.sqrt(f);
    }

    private static float exp(float f) {
        return (float) Math.exp(f);
    }
}
