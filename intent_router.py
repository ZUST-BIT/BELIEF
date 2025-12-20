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
            model_name='gpt-5', 
            temperature=0,
            api_key=args.api_key_gpt, 
            base_url=args.api_url_gpt,
            model_kwargs={"response_format": {"type": "json_object"}}
        )

    def intent_router(self, question, context=""):
        """
        Args:
            判断是否需要检索。
            question: 用户的提问
            context: 用户提供的背景文本（如果有）
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", routing_prompt),
            ("user", "User Context: {context}\nUser Query: {question}")
        ])

        chain = prompt | self.llm | StrOutputParser()

        try:
            response_str = chain.invoke({
                "question": question,
                "context": context if context else "No context provided."
            })

            # 清洗与解析
            cleaned_res = response_str.strip()
            if cleaned_res.startswith("```json"):
                cleaned_res = cleaned_res.replace("```json", "").replace("```", "")
            
            res_dict = json.loads(cleaned_res)
            if "extracted_entities" not in res_dict:
                res_dict["extracted_entities"] = []
                
            return res_dict

        except Exception as e:
            print(f"❌ Intent Router Error: {e}")
            fallback_need = "No" if context and len(context) >50 else "Yes"
            # 保底逻辑：如果解析失败，默认假设需要检索
            return {
                "analysis": "Error in parsing, fallback to search.",
                "need_retrieval": fallback_need,
                "rewritten_query": question, 
                "extracted_entities": []
            }

# if __name__ == '__main__':
#     router = IntentRouter()
    
    # 测试 Case 1: 开放问题（无上下文）-> 预期 need_retrieval: YES
    # q1 = "We investigated the actual role of MRI versus arthroscopy in the detection and characterization of occult bone and/or cartilage injuries in patients with previous musculoskeletal trauma of the knee, pain and severe functional impairment. Occult post-traumatic osteochondral injuries of the knee are trauma-related bone and/or cartilage damage missed at plain radiography.\nWe retrospectively selected 70 patients (men:women = 7:3; age range: 35 +/- 7 years) with a history of acute musculoskeletal trauma, negative conventional radiographs, pain and limited joint movements. All patients were submitted to conventional radiography, arthroscopy and MRI, the latter with 0.5 T units and T1-weighted SE. T2-weighted GE and FIR sequences with fat suppression.\nWe identified three types of occult post-traumatic injuries by morpho-topographic and signal intensity patterns: bone bruises (no. 25), subchondral (no. 33) and osteochondral (no. 35) injuries. Arthroscopy depicted 45 osteochondral and 19 chondral injuries. A bone bruise was defined as a typical subcortical area of signal loss, with various shapes, on T1-weighted images and of increased signal intensity on T2-weighted and FIR images. The cortical bone and articular cartilage were normal in all cases, while osteochondral injuries exhibited associated bone and cartilage damage with the same abnormal MR signal intensity. Sprain was the mechanism of injury in 52 cases, bruise in 12 and stress in 6. In 52 sprains (30 in valgus), the injury site was the lateral compartment in 92.3% of cases (100% in valgus), associated with meniscal damage in 73% of cases (90% in valgus) and with ligament injury in 90.4% (100% in valgus). In 12 bruises, the injury site was the lateral compartment in 58.3% of cases, the knee cap in 25% and the medial compartment in 16.7%; meniscal damage was associated in 25% of cases and ligament damage in 8.3%. In 6 stress injuries, the injury site was localized in the medial tibial condyle in 80% of cases, while meniscal and ligament tears were absent.\n\"Occult\" posttraumatic lesions of the knee: can magnetic resonance substitute for diagnostic arthroscopy?"
    # c1 = "" # 无上下文
    # print("--- Case 1 Result ---")
    # res = router.intent_router(q1, c1)
    # print(json.dumps(res, indent=2, ensure_ascii=False))
    # print("\n")
    # print(res.get("need_retrieval", "N/A"))
    # print(res.get("rewritten_query", "N/A"))
    # print(res.get("extracted_entities", "N/A"))
    # print("\n")
    
    # 测试 Case 2: 阅读理解（有上下文）-> 预期 need_retrieval: NO
    # q2 = "针对Lenvatinib的药物敏感性来说，结合组学的信息，哪类特征（包括基因突变，包括临床特征）的病人更容易出现对Lenvatinib的耐药现象，这类病人在基因组的突变或者转录组的差异基因上有什么特征。"
    # c2 = ""
    # print("\n--- Case 2 Result ---")
    # print(json.dumps(router.intent_router(q2, c2), indent=2, ensure_ascii=False))