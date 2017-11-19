package ololo.ftrl;


import com.google.common.io.Closer;

import java.io.*;

/**
 * https://github.com/alexeygrigorev/outbrain-click-prediction-kaggle/blob/master/ftrl.py
 */
public class FtrlProximalModel implements Serializable {

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

    public FtrlProximalModel(float alpha, float beta, float l1, float l2, int numFeatures,
                             float[] n, float[] z, float[] w,
                             float nIntercept, float zIntercept, float wIntercept) {
        this.alpha = alpha;
        this.beta = beta;
        this.l1 = l1;
        this.l2 = l2;
        this.numFeatures = numFeatures;
        this.n = n;
        this.z = z;
        this.w = w;
        this.nIntercept = nIntercept;
        this.zIntercept = zIntercept;
        this.wIntercept = wIntercept;
    }

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

    public static float logloss(float p, float y) {
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

    public static FtrlProximalModel load(String path) throws IOException {
        Closer closer = Closer.create();

        InputStream is = closer.register(new FileInputStream(new File(path)));
        InputStream bis = closer.register(new BufferedInputStream(is));
        DataInputStream dis = closer.register(new DataInputStream(bis));

        float alpha = dis.readFloat();
        float beta = dis.readFloat();
        float l1 = dis.readFloat();
        float l2 = dis.readFloat();

        int numFeatures = dis.readInt();

        float[] n = readFloatArray(dis, numFeatures);
        float[] z = readFloatArray(dis, numFeatures);
        float[] w = readFloatArray(dis, numFeatures);

        float nIntercept = dis.readFloat();
        float zIntercept = dis.readFloat();
        float wIntercept = dis.readFloat();

        closer.close();

        return new FtrlProximalModel(alpha, beta, l1, l2, numFeatures,
                n, z, w, nIntercept, zIntercept, wIntercept);
    }


    public void save(String path) throws IOException {
        Closer closer = Closer.create();

        FileOutputStream fos = closer.register(new FileOutputStream(new File(path)));
        BufferedOutputStream bos = closer.register(new BufferedOutputStream(fos));
        DataOutputStream dos = closer.register(new DataOutputStream(bos));

        dos.writeFloat(alpha);
        dos.writeFloat(beta);
        dos.writeFloat(l1);
        dos.writeFloat(l2);
        dos.writeInt(numFeatures);

        writeFloatArray(dos, n);
        writeFloatArray(dos, z);
        writeFloatArray(dos, w);

        dos.writeFloat(nIntercept);
        dos.writeFloat(zIntercept);
        dos.writeFloat(wIntercept);

        closer.close();
    }

    private static void writeFloatArray(DataOutputStream dos, float[] array) throws IOException {
        for (int i = 0; i < array.length; i++) {
            dos.writeFloat(array[i]);
        }
    }

    private static float[] readFloatArray(DataInputStream dis, int n) throws IOException {
        float[] array = new float[n];
        for (int i = 0; i < n; i++) {
            array[i] = dis.readFloat();
        }
        return array;
    }
}
