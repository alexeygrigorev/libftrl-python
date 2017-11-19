package ololo.ftrl.ololo.data;

import com.google.common.collect.AbstractIterator;
import com.google.common.collect.Iterators;
import com.google.common.collect.PeekingIterator;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.function.Function;

public class GroupingIterator<K, E> extends AbstractIterator<Group<K, E>> {

    private PeekingIterator<E> iterator;
    private Function<E, K> key;

    public GroupingIterator(Iterator<E> iterator, Function<E, K> key) {
        this.iterator = Iterators.peekingIterator(iterator);
        this.key = key;
    }

    @Override
    protected Group<K, E> computeNext() {
        if (!iterator.hasNext()) {
            return endOfData();
        }

        E first = iterator.next();
        K groupKey = key.apply(first);
        List<E> group = new ArrayList<>();
        group.add(first);

        while (iterator.hasNext()) {
            E possiblyNext = iterator.peek();
            Object nextItemGroupKey = key.apply(possiblyNext);

            if (groupKey.equals(nextItemGroupKey)) {
                group.add(iterator.next());
            } else {
                break;
            }
        }

        return new Group<>(groupKey, group);
    }
}
