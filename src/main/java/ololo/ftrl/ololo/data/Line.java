package ololo.ftrl.ololo.data;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.builder.ReflectionToStringBuilder;
import org.apache.commons.lang3.builder.ToStringStyle;

import java.util.Arrays;

public class Line {

    private int groupId;
    private int f0;
    private int f1;
    private int[] features;
    private byte[] values;
    private byte click;
    private float propensity;

    public static Line parse(String line, Line result, boolean skipUnlabeled) {
        String[] split = StringUtils.split(line, '|');

        int groupId = Integer.parseInt(split[0].trim());
        result.setGroupId(groupId);

        if (split.length == 4) {
            byte label = parseLabel(split[1]);
            result.setClick(label);

            float propensity = parsePropensity(split[2]);
            result.setPropensity(propensity);

            parseFeatures(split[3], result);

            return result;
        } else if (split.length == 2) {
            if (skipUnlabeled) {
                return null;
            }

            result.setClick((byte) -1);
            result.setPropensity(Float.NaN);
            parseFeatures(split[1], result);
            return result;
        }

        throw new IllegalArgumentException("unexpected input: " + Arrays.toString(split));
    }

    private static void parseFeatures(String line, Line result) {
        line = StringUtils.stripStart(line,"f ").trim();
        String[] split = line.split(" ");

        String f0Str = split[0].trim();
        int f0 = Integer.parseInt(f0Str.substring(2));

        String f1Str = split[1].trim();
        int f1 = Integer.parseInt(f1Str.substring(2));

        result.setF0(f0);
        result.setF1(f1);

        int size = split.length;
        int[] features = new int[size - 2];
        byte[] values = new byte[size - 2];

        for (int i = 2; i < split.length; i ++) {
            String s = split[i];
            String[] featureSplit = s.split(":");
            features[i - 2] = Integer.parseInt(featureSplit[0]) - 2;
            values[i - 2] = Byte.parseByte(featureSplit[1]);
        }

        result.setFeatures(features);
        result.setValues(values);
    }

    private static float parsePropensity(String p) {
        String propensity = StringUtils.stripStart(p, "p ").trim();
        return Float.parseFloat(propensity);
    }

    private static byte parseLabel(String l) {
        String label = StringUtils.stripStart(l, "l ").trim();
        if ("0.999".equals(label)) {
            return 0;
        } else if ("0.001".equals(label)) {
            return 1;
        } else {
            throw new IllegalArgumentException("unexpected label string: " + l);
        }
    }

    public int getGroupId() {
        return groupId;
    }

    public int getF0() {
        return f0;
    }

    public int getF1() {
        return f1;
    }

    public int[] getFeatures() {
        return features;
    }

    public byte[] getValues() {
        return values;
    }

    public byte getClick() {
        return click;
    }

    public float getPropensity() {
        return propensity;
    }

    public void setGroupId(int groupId) {
        this.groupId = groupId;
    }

    public void setF0(int f0) {
        this.f0 = f0;
    }

    public void setF1(int f1) {
        this.f1 = f1;
    }

    public void setFeatures(int[] features) {
        this.features = features;
    }

    public void setValues(byte[] values) {
        this.values = values;
    }

    public void setClick(byte click) {
        this.click = click;
    }

    public void setPropensity(float propensity) {
        this.propensity = propensity;
    }

    @Override
    public String toString() {
        return ReflectionToStringBuilder.toString(this, ToStringStyle.NO_CLASS_NAME_STYLE);
    }
}