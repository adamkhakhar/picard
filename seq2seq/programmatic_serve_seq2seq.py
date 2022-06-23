import requests


class Picard:
    """
    Interact with picard model
    """

    def __init__(self, port=8000):
        self.port = port

    def query_model(self, db_id, question):
        question_param_string = ""
        for word in question.split():
            question_param_string += word + "%20"
        question_param_string = question_param_string[:-3]
        return requests.get(f"http://localhost:{self.port}/ask/{db_id}/{question_param_string}").json()
