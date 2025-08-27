class Node:
    __slots__ = 'prev', 'next', 'key', 'value'

    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value


class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.dummy = Node()
        self.dummy.prev = self.dummy
        self.dummy.next = self.dummy
        self.key_to_node = {}

    def get(self, key: int) -> int:
        node = self.get_node(key)
        return node.value if node else -1

    def get_node(self, key):
        if key not in self.key_to_node:
            return None
        node = self.key_to_node[key]
        self.remove(node)
        self.put_front(node)
        return node

    def put_front(self, node):
        node.prev = self.dummy
        node.next = self.dummy.next
        node.prev.next = node
        node.next.prev = node

    def remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def put(self, key: int, value: int) -> None:
        node = self.get_node(key)
        if node:
            node.value = value
            return
        self.key_to_node[key] = node = Node(key, value)
        self.put_front(node)
        if len(self.key_to_node) > self.capacity:
            last_node = self.dummy.prev
            del self.key_to_node[last_node.key]
            self.remove(last_node)

# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
# 初始化 LRU 缓存，容量为 3
cache = LRUCache(3)

# 添加一些数据
cache.put(1, 1)  # cache: {1=1}
cache.put(2, 2)  # cache: {1=1, 2=2}
cache.put(3, 3)  # cache: {1=1, 2=2, 3=3}

# 访问键 2
print(cache.get(2))  # 返回 2, cache: {1=1, 3=3, 2=2}， 2 成为最近访问

# 添加一个新数据项，缓存已经满了，删除最久未使用的项（键 1）
cache.put(4, 4)  # cache: {3=3, 2=2, 4=4}

# 访问键 1（应该不存在）
print(cache.get(1))  # 返回 -1，表示不存在

# 访问键 3
print(cache.get(3))  # 返回 3, cache: {2=2, 4=4, 3=3}，3 成为最近访问

# 再次添加一个新数据项，删除最久未使用的项（键 2）
cache.put(5, 5)  # cache: {4=4, 3=3, 5=5}

# 访问键 2（应该不存在）
print(cache.get(2))  # 返回 -1，表示不存在

# 打印最终缓存内容
# cache: {4=4, 3=3, 5=5}
