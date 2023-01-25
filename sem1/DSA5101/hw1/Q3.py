from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference


class Q3Solution:
    @staticmethod
    def solve():
        file_a = open("fileA.txt", "r")
        str_a = file_a.read()
        file_a.close()

        file_b = open('fileB.txt', 'r')
        str_b = file_b.read()
        file_b.close()

        punctuation = ['.', '?', '!', ',', ':', ';', '/', '(', ')', '...', '@', "'", '"']
        for punc in punctuation:
            str_a = str_a.replace(punc, '')
            str_b = str_b.replace(punc, '')
        str_a = str_a.lower()
        str_b = str_b.lower()

        uniq_words_in_a_b = list(set(str_a.split()).intersection(set(str_b.split())))

        wc = Q3Solution.word_count(str_a, str_b, uniq_words_in_a_b)

        Q3Solution.plot_bar_chart(wc)

    @staticmethod
    def word_count(a, b, uniq):
        word_count_a = {}
        word_count_b = {}
        for word in uniq:
            if a.count(word) > b.count(word):
                word_count_a[word] = a.count(word)
                word_count_b[word] = b.count(word)
        sort_word_count_a = dict(sorted(word_count_a.items(), key=lambda x: x[1], reverse=True))
        merged_word_count = {}
        for key in sort_word_count_a.keys():
            merged_word_count[key] = [word_count_a[key], word_count_b[key]]
        return merged_word_count

    @staticmethod
    def plot_bar_chart(word_count):
        wb = Workbook()
        ws = wb.active
        ws.title = 'Q3'

        headings = ['Word', '#(Occurrences) in A', '#(Occurrences) in B']
        ws.append(headings)

        for word in word_count:
            ws.append([word] + word_count[word])

        # Data for plotting
        values = Reference(ws, min_col=2, max_col=3, min_row=1, max_row=6)

        cats = Reference(ws, min_col=1, max_col=1, min_row=2, max_row=6)

        # Create object of BarChart class
        chart = BarChart()
        chart.add_data(values, titles_from_data=True)
        chart.set_categories(cats)

        # set titles
        chart.title = "Word Count of A and B"
        chart.x_axis.title = "Word"
        chart.y_axis.title = "Word Count"

        ws.add_chart(chart, "G2")

        wb.save('WordOccurence.xlsx')


if __name__ == "__main__":
    Q3Solution.solve()
