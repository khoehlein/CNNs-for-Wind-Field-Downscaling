import sys


class ProgressBar:
    def __init__(self, numCounts, charLength=30, displaySumCount=False):
        self.numCounts = numCounts
        self.charLength = charLength
        self.displaySummary = displaySumCount

    def proceed(self, i):
        progress = i / float(self.numCounts)
        barProgress = int(self.charLength * progress)

        summary = ''
        if self.displaySummary:
            summary = "{}/{}".format(i, self.numCounts)

        sys.stdout.write(
            '\rProgress: |{}{}{}{}| {}% ({}){}'.format(
                '=' * max(barProgress - 1, 0),
                '>' * int(0 < progress < 1),
                '=' * int(progress == 1),
                '.' * (self.charLength - barProgress),
                int(progress * 100),
                summary,
                '\n' * int(progress == 1)
            )
        )
        sys.stdout.flush()
