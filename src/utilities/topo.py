def topo_reverse_sort(node):
    visited = set()
    res = []

    def dfs(node):
        if node not in visited:
            visited.add(node)
            for parent in node._parents:
                dfs(parent)
            res.append(node)

    dfs(node)
    return res
