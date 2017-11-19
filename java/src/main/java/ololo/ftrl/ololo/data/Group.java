package ololo.ftrl.ololo.data;

import java.util.List;

public class Group<K, V> {

    private final K key;
    private final List<V> group;

    public Group(K key, List<V> group) {
        this.key = key;
        this.group = group;
    }

    public K getKey() {
        return key;
    }

    public List<V> getGroup() {
        return group;
    }
}
