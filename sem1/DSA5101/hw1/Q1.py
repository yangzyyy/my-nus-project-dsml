class Q1Solution:
    @staticmethod
    def solve():
        f = open("Two_strings.txt", "r")
        x = f.readline()
        y = f.readline()
        f.close()

        m = len(x)
        n = len(y)

        lookup = [[0 for x in range(n + 1)] for y in range(m + 1)]
        lookup = Q1Solution.scs_length(x, y, m, n, lookup)
        return Q1Solution.scs_dp(x, y, m, n, lookup)

    @staticmethod
    def scs_length(x, y, m, n, lookup):
        for i in range(m + 1):
            lookup[i][0] = i

        for j in range(n + 1):
            lookup[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    lookup[i][j] = lookup[i - 1][j - 1] + 1
                else:
                    lookup[i][j] = min(lookup[i - 1][j] + 1, lookup[i][j - 1] + 1)
        return lookup

    @staticmethod
    def scs_dp(str1, str2, m, n, lookup):
        if m == 0:
            return str2
        if n == 0:
            return str1

        if str1[m - 1] == str2[n - 1]:
            return Q1Solution.scs_dp(str1[0:m - 1], str2[0:n - 1], m - 1, n - 1, lookup) + str1[m - 1]

        else:
            if lookup[m - 1][n] < lookup[m][n - 1]:
                return Q1Solution.scs_dp(str1, str2, m - 1, n, lookup) + str1[m - 1]
            else:
                return Q1Solution.scs_dp(str1, str2, m, n - 1, lookup) + str2[n - 1]


if __name__ == "__main__":
    print(Q1Solution.solve())
