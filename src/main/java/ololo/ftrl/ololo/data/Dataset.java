package ololo.ftrl.ololo.data;

import com.google.common.collect.AbstractIterator;
import com.google.common.io.Closer;
import org.apache.commons.io.LineIterator;
import org.apache.commons.lang3.StringUtils;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.zip.GZIPInputStream;

public class Dataset extends AbstractIterator<Line> implements AutoCloseable {

    private final Line line = new Line();
    private final Closer closer;
    private final LineIterator lines;

    private final boolean skipUnlabeled;

    public Dataset(String path, boolean skipUnlabeled) throws IOException {
        this.skipUnlabeled = skipUnlabeled;

        Closer closer = Closer.create();

        InputStream is = closer.register(new FileInputStream(new File(path)));
        GZIPInputStream gz = closer.register(new GZIPInputStream(is, 8192));
        Reader reader = closer.register(new InputStreamReader(gz, StandardCharsets.UTF_8));

        LineIterator li = closer.register(new LineIterator(reader));

        this.closer = closer;
        this.lines = li;
    }

    @Override
    protected Line computeNext() {
        try {
            return nextUnsafe();
        } catch (IOException e) {
            throw new RuntimeException("we're screwed (most likely)");
        }
    }

    private Line nextUnsafe() throws IOException {
        while (lines.hasNext()) {
            String next = lines.next();
            if (StringUtils.isBlank(next)) {
                continue;
            }

            Line result = Line.parse(next, line, skipUnlabeled);
            if (result == null) {
                continue;
            }

            return result;
        }

        closer.close();
        return endOfData();
    }

    @Override
    public void close() throws Exception {
        closer.close();
    }

    public static void main(String[] args) throws Exception {
        String path = "/home/alexey/tmp/criteo/criteo_train_small.txt.gz";

        try (Dataset lines = new Dataset(path, true)) {
            while (lines.hasNext()) {
                System.out.println(lines.next());
            }
        }

    }
}
