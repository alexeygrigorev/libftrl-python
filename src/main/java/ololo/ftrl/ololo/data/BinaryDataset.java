package ololo.ftrl.ololo.data;

import com.google.common.collect.AbstractIterator;
import com.google.common.io.Closer;

import java.io.*;
import java.util.concurrent.Callable;
import java.util.function.Function;

public class BinaryDataset extends AbstractIterator<Line> implements AutoCloseable {

    private final DataInputStream dis;
    private final Closer closer;
    private final Callable<Line> lineFactory;

    public BinaryDataset(String name) throws IOException {
        this(name, false);
    }

    public BinaryDataset(String name, boolean createNewObjects) throws IOException {
        Closer closer = Closer.create();

        InputStream is = closer.register(new FileInputStream(new File(name)));
        InputStream bis = closer.register(new BufferedInputStream(is));
        DataInputStream dis = closer.register(new DataInputStream(bis));

        this.dis = dis;
        this.closer = closer;

        if (createNewObjects) {
            lineFactory = () -> new Line();
        } else {
            Line line = new Line();
            lineFactory = () -> line;
        }
    }

    @Override
    protected Line computeNext() {
        try {
            return nextUnsafe();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private Line nextUnsafe() throws Exception {
        if (dis.available() == 0) {
            closer.close();
            return endOfData();
        }

        Line data = lineFactory.call();
        data.read(dis);
        return data;
    }

    @Override
    public void close() throws Exception {
        closer.close();
    }
}
