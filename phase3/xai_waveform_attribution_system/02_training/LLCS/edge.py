class Edge:
    def __init__(self, node, age = 1, weight=1):
        self.node = node
        self.weight = weight
        self.age = age

    def get_node(self):
        return self.node
