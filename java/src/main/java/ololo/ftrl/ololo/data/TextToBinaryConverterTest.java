package ololo.ftrl.ololo.data;

import com.google.common.base.Stopwatch;
import com.google.common.io.Closer;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;

public class TextToBinaryConverterTest {

    public static void main(String[] args) throws Exception {
        String inputPath = "/home/alexey/tmp/criteo/criteo_test_release.txt.gz";
        String outputPath = "/home/alexey/tmp/criteo/criteo_test_all.bin";

        boolean skipUnlabeled = false;

        if (new File(outputPath).exists()) {
            throw new IllegalArgumentException(String.format("file %s already exists", outputPath));
        }

        Stopwatch stopwatch = Stopwatch.createStarted();

        Closer closer = Closer.create();
        FileOutputStream fos = closer.register(new FileOutputStream(new File(outputPath)));
        BufferedOutputStream bos = closer.register(new BufferedOutputStream(fos));
        DataOutputStream out = closer.register(new DataOutputStream(bos));

        int line = 0;
        try (TextDataset lines = new TextDataset(inputPath, skipUnlabeled)) {
            while (lines.hasNext()) {
                Line next = lines.next();
                next.write(out);
                line++;

                if (line % 10000 == 0) {
                    System.out.println("processed line #" + line);
                }
            }
        }

        closer.close();

        System.out.printf("finished writing from %s to %s in %s%n", inputPath, outputPath, stopwatch.stop());
    }
}
