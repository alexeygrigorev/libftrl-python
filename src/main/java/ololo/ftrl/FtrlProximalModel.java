package ololo.ftrl;


/**
 * https://github.com/alexeygrigorev/outbrain-click-prediction-kaggle/blob/master/ftrl.py
 */
public class FtrlProximalModel {

    private static final float TOLERANCE = 1e-6f;

    private final float alpha;
    private final float beta;
    private final float l1;
    private final float l2;
    private final int numFeatures;

    private final float[] n;
    private final float[] z;
    private final float[] w;

    private float nIntercept;
    private float zIntercept;
    private float wIntercept;

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

    public static float sigma(float n, float grad, float alpha) {
        return (sqrt(n + grad * grad) - sqrt(n)) / alpha;
    }

    public float fit(int[] values, float y) {
        float p = predict(values);
        float grad = p - y;

        float sigmaIntercept = sigma(nIntercept, grad, alpha);
        zIntercept = zIntercept + grad - sigmaIntercept * wIntercept;
        nIntercept = nIntercept + grad * grad;

        for (int i : values) {
            float sigma = sigma(n[i], grad, alpha);
            z[i] = z[i] + grad - sigma * w[i];
            n[i] = n[i] + grad * grad;
        }

        return logloss(p, y);
    }

    private static float logloss(float p, float y) {
        if (y == 1.0f) {
            return -log(Math.max(p, TOLERANCE));
        } else if (y == 0.0f) {
            return -log(Math.min(1 - p, 1 - TOLERANCE));
        }
        throw new IllegalArgumentException("expect 1 or 0, got " + y);
    }

    public static float calculateW(float z, float n, float l1, float l2, float alpha, float beta) {
        float sign = sign(z);
        if (sign * z <= l1) {
            return 0.0f;
        }

        float w = (sign * l1 - z) / ((beta + sqrt(n)) / alpha + l2);

        return w;
    }

    public float predict(int[] values) {
        wIntercept = calculateW(zIntercept, nIntercept, l1, l2, alpha, beta);
        float wtx = wIntercept;

        for (int i : values) {
            w[i] = calculateW(z[i], n[i], l1, l2, alpha, beta);
            wtx = wtx + w[i];
        }

        return sigmoid(wtx);
    }

    private static float sigmoid(float x) {
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

    private static float log(float f) {
        return (float) Math.log(f);
    }

    private static float sign(float f) {
        if (f < 0) {
            return -1.0f;
        } else {
            return 1.0f;
        }
    }
}
