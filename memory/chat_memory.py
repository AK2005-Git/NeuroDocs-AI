class ChatMemory:

    def __init__(self):
        self.history = []

    def add_user(self, text):

        self.history.append({
            "role": "user",
            "content": text
        })

    def add_assistant(self, text):

        self.history.append({
            "role": "assistant",
            "content": text
        })

    def get_recent(self, k=5):

        return self.history[-k:]

    def clear(self):

        self.history = []