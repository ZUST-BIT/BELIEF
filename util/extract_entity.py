from openai import OpenAI

client = OpenAI(
    api_key = 'sk-0f17b61caf3f48e99944865634bd3a1c',
    base_url = 'https://api.deepseek.com'
)

def extract_entity(query):
    """
    使用 llm 模型提取实体
    """
    from prompt import extr_norm_prompt
    prompt = extr_norm_prompt + f"\n问题如下:{query}"
    response = client.chat.completions.create(
        model = 'deepseek-chat',
        messages =[
            {"role":"system","content":"You are a helpful AI assistant for entity extraction."},
            {"role":"user","content":prompt},
        ],
        stream=False,
    )
    res = response.choices[0].message.content
    res = norm_llm_output(res)
    return res

def norm_llm_output(res):
    """ 规范化llm的输出 """
    if res.startswith("```"):
        first_newline = res.find("\n")
        if first_newline != -1:
            res = res[first_newline + 1:]
    if res.endswith("```"):
        res = res[:-3]
    res = res.strip()
    try:
        import ast
        res_dict = ast.literal_eval(res)
    except Exception as e:
        print(f"Error: {e}")
    return res_dict


# query = '针对Lenvatinib的药物敏感性来说，结合组学的信息，哪类特征（包括基因突变，包括临床特征）的病人更容易出现对Lenvatinib的耐药现象，这类病人在基因组的突变或者转录组的差异基因上有什么特征。' 
# res = extract_entity(query)
# print(res)