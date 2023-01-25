class Q1Solution:

    @staticmethod
    def solve():
        graph = []
        out_notes = []
        in_notes = []
        f = open("graph.txt", "r")
        for line in f:
            line = line.split(',')
            out_notes.append(line[0])
            in_notes.append(line[1].strip('\n'))
        graph.append(out_notes)
        graph.append(in_notes)
        f.close()

        result = Q1Solution.topological_sort(graph, out_notes, in_notes)
        f = open('topological_sort.txt', 'w')
        print(*result, sep=",", file=f)
        f.close()

    @staticmethod
    def topological_sort(graph, out_notes, in_notes):
        sorted_elements = []
        nodes_without_in_edges = []
        for node in set(out_notes):
            if node not in in_notes:
                nodes_without_in_edges.append(node)

        while len(nodes_without_in_edges) > 0:
            temp_out_node = nodes_without_in_edges[-1]
            nodes_without_in_edges.pop()
            sorted_elements.append(temp_out_node)
            temp_in_node = []
            match_index = [i for i in range(len(graph[0])) if graph[0][i] == temp_out_node]
            for i in match_index:
                temp_in_node.append(graph[1][i])
            for i in sorted(match_index, reverse=True):
                del graph[0][i]
                del graph[1][i]
            for node in temp_in_node:
                if Q1Solution.check_no_incoming_edge(node, graph):
                    nodes_without_in_edges.append(node)
        if len(graph[0]) != 0 or len(graph[1]) != 0:
            return 'The graph has at least one circle!'
        else:
            return sorted_elements

    @staticmethod
    def check_no_incoming_edge(node, graph):
        if node not in graph[1]:
            return True
        else:
            return False


if __name__ == "__main__":
    Q1Solution.solve()
