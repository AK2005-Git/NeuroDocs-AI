class Analytics:

    def __init__(self):

        self.total_queries = 0

    def increment(self):

        self.total_queries += 1

    def get_total_queries(self):

        return self.total_queries