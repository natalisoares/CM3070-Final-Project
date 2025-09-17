import base64
from openai import OpenAI

def get_openai_client():
    encoded_key = "c2stcHJvai1Fa3o0VVZDU3doZU5vT3pianBlejdYekE0Q3BlVVJ0RFlIQmw3anFqTDFJNjlxd2pmSDlqTWlhZnRpQXdpSVg0RUdxbW1QaV96NlQzQmxia0ZKald1NC1ndXFnRVdWYzFSWFJSSFlVUlZVZG5XamZXR090dFkyTVBqRmRrNHFBSlFGSEVoaUpqSEZSZG1zU0owbXZ2aDBxcFJRZ0E="
    api_key = base64.b64decode(encoded_key).decode()
    return OpenAI(api_key=api_key)
