def all_column(matrix):
    columns = []
    n = len(matrix[0])

    for i in range(n):
        column = [row[i] for row in matrix]
        columns.append(column)

    return columns


def mul(matrix1, matrix2):
    m, n = len(matrix1[0]), len(matrix2)

    if m != n:
        return []

    output = []
    for row in (matrix1):
        new_row = []
        for col in all_column(matrix2):
            num_elements = len(col)
            sum = 0

            for i in range(num_elements):
                sum = sum + row[i] * col[i]
            new_row.append(sum)

        output.append(new_row)

    return output


A = [[1, 0, 0],
     [0, 0, 3],
     [0, 2, 0]]

B = [[1, 1],
     [0, .5],
     [2, 1 / 3.0]]

C = [[1, 0, 0],
     [0, 0, 0.5],
     [0, 1 / 3.0, 0]]

print(mul(A, B))
print(mul(B,A))
print(mul(A, C))
