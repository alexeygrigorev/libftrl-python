package ololo.ftrl;

import com.google.common.base.Stopwatch;
import com.google.common.collect.Iterators;
import com.google.common.io.Closer;
import com.google.common.primitives.Floats;
import ololo.ftrl.ololo.data.BinaryDataset;
import ololo.ftrl.ololo.data.Group;
import ololo.ftrl.ololo.data.GroupingIterator;
import ololo.ftrl.ololo.data.Line;
import org.apache.commons.io.output.NullOutputStream;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.zip.GZIPOutputStream;

public class ModelTest {
    public static void main(String[] args) throws Exception {
        String modelPath = "model_n=6_a=1.000_b=1.000_l1=20.000_l2=0.000.bin";
        FtrlProximalModel model = FtrlProximalModel.load(modelPath);

        String testPath = "/home/alexey/tmp/criteo/criteo_test_all.bin";

        int cnt = 0;

        String output = "/home/alexey/tmp/criteo/pred_out.txt.gz";

        Closer closer = Closer.create();
        FileOutputStream fos = closer.register(new FileOutputStream(new File(output)));
        GZIPOutputStream gzip = closer.register(new GZIPOutputStream(fos, 8192));
        PrintWriter pw = closer.register(new PrintWriter(gzip));

        Stopwatch pass = Stopwatch.createStarted();
        try (BinaryDataset lines = new BinaryDataset(testPath, true)) {
            GroupingIterator<Integer, Line> groups = new GroupingIterator<>(lines, l -> l.getGroupId());

            while (groups.hasNext()) {
                Group<Integer, Line> group = groups.next();
                List<Line> elements = group.getGroup();

                List<String> preds = new ArrayList<>(elements.size());

                for (int i = 0; i < elements.size(); i++) {
                    Line line = elements.get(i);
                    int[] features = line.getFeatures();
                    float pred = model.predict(features);
                    preds.add(String.format("%d:%f", i, pred));
                }

                String join = String.join(",", preds);
                pw.print(group.getKey());
                pw.print(';');
                pw.print(join);
                pw.println();

                cnt++;

                if (cnt % 10000 == 0) {
                    System.out.println("processed " + cnt + " groups so far");
                }
            }
        }

        pw.flush();
        closer.close();

        System.out.printf("pass finished in %s%n", pass.stop());
    }

}
