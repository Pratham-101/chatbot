# Minimal stub for response_quality.py to allow backend startup

class ResponseQuality:
    def __init__(self, accuracy=10, completeness=10, clarity=10, relevance=10, overall_score=10, feedback="Stub quality"): 
        self.accuracy = accuracy
        self.completeness = completeness
        self.clarity = clarity
        self.relevance = relevance
        self.overall_score = overall_score
        self.feedback = feedback
    def dict(self):
        return self.__dict__

class StructuredResponse:
    def __init__(self):
        self.summary = "No summary available."
        self.key_points = []
        self.fund_details = {}
        self.performance_data = {}
        self.risk_metrics = {}
        self.recommendations = []
        self.disclaimer = "No disclaimer."
        self.sources = []
    def dict(self):
        return self.__dict__

def response_evaluator(*args, **kwargs):
    return ResponseQuality()

def format_structured_response(structured_response, raw_response=None):
    if raw_response:
        return raw_response
    return "Stub structured response"

class StructuredGenerator:
    async def generate_structured_response(self, query, raw_response, real_time_data):
        return StructuredResponse()
    def format_structured_response(self, structured_response, raw_response=None):
        return format_structured_response(structured_response, raw_response)

structured_generator = StructuredGenerator() 