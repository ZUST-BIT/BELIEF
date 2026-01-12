import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import set_argument
from prompt import routing_prompt

args = set_argument()

class IntentRouter:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name='gpt-4o-mini', 
            temperature=0,
            api_key=args.api_key_gpt, 
            base_url=args.api_url_gpt,
            model_kwargs={"response_format": {"type": "json_object"}}
        )

        self.kg_schema = """
        # PrimeKG Schema: 
        * **Node Types**: 'GeneProtein', 'Drug', 'EffectPhenotype', 'Disease', 'BiologicalProcess', 'MolecularFunction', 'Pathway', 'Exposure', 'CellularComponent', 'Anatomy'.
        * **Key Relations**: 'ProteinProtein', 'Target', 'Enzyme', 'Carrier', 'Transporter', 'Contraindication', 'Indication', 'OffLabelUse', 'SynergisticInteraction', 'ParentChild', 'AssociatedWith', 'PhenotypePresent', 'PhenotypeAbsent', 'SideEffect', 'InteractsWith', 'LinkedTo', 'ExpressionPresent', 'ExpressionAbsent'.
        """

    def intent_router(self, question, context=""):
        """
        Args:
            判断是否需要检索。
            question: 用户的提问
            context: 用户提供的背景文本（如果有）
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", routing_prompt),
            ("user", "PrimeKG Schema:{kg_schema_info}\nUser Query: {question}\nContext: {context}")
        ])
        # print(f"提示词是：{prompt}")
        chain = prompt | self.llm | StrOutputParser()

        try:
            response_str = chain.invoke({
                "question": question,
                "kg_schema_info": self.kg_schema,
                "context": context if context else "No context provided."
            })

            # 清洗与解析
            cleaned_res = response_str.strip()
            if cleaned_res.startswith("```json"):
                cleaned_res = cleaned_res.replace("```json", "").replace("```", "")
            elif cleaned_res.startswith("```"):
                cleaned_res = cleaned_res.replace("```", "")
            
            res_dict = json.loads(cleaned_res)
            if "rewritten_query" not in res_dict:
                res_dict["rewritten_query"] = question
                
            return res_dict

        except Exception as e:
            print(f"❌ Intent Router Error: {e}")
            # 保底逻辑：如果解析失败，默认假设需要检索
            return {
                "resoning": "Error in parsing, fallback to search.",
                "question_entities": [],
                "answer_entities": [], 
                "rewritten_query": question, 
            }