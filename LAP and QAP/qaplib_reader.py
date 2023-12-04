def read_integers(filename):
    with open(filename) as f:
        return [int(elem) for elem in f.read().split()]

def read_qap(filename):
    file_it = iter(read_integers(filename))

    # Number of points
    n = next(file_it)

    # Distance between locations
    A = [[next(file_it) for j in range(n)] for i in range(n)]
    # Flow between factories
    B = [[next(file_it) for j in range(n)] for i in range(n)]
    
    return A,B

def read_qap_solution(filename):
    with open(filename) as f:
        lines = f.readlines()

    # optimal energy
    e = lines[0].split()
    e = e[1]
    e = int(e)
    
    return e