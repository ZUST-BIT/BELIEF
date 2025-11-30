from openai import OpenAI
import ast
from config import set_argument
from prompt import routing_prompt
from neo4j import Neo4jManager
args = set_argument()
class IntentRouter:
    def __init__(self):
        self.client = OpenAI(api_key=args.api_key,base_url=args.api_url)
    def intent_router(self, question):
        kg_schema = Neo4jManager().get_kg_schema()
        prompt = routing_prompt + f"\nQuestion: {question}\nKG Schema: {kg_schema}\n"
        response = self.client.chat.completions.create(
            model='deepseek-chat',
            messages=[
                {'role':"user",'content':prompt}
            ], 
            stream=False
        )
        res = response.choices[0].message.content
        res = ast.literal_eval(res)
        return res


