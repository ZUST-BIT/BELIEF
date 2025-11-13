from openai import OpenAI

client = OpenAI(
    api_key = 'sk-0f17b61caf3f48e99944865634bd3a1c',
    base_url = 'https://api.deepseek.com'
)
def answer(graph_data,pubmed_data,pubmed_data_bm25,omics_data,query):
    """
    使用 llm 模型提取实体
    """
    data = f"""请基于以下信息综合回答用户问题。信息包括：

    知识图谱路径信息:{graph_data}

    文献信息：{pubmed_data} + {pubmed_data_bm25}

    病例样本信息：{omics_data}
    用户问题是：{query}
    请以生物医学研究人员的身份回答，内容要求科学严谨、逻辑清晰、层次分明。"""
    from prompt import answer_prompt
    prompt = answer_prompt + f"\n示例提示:{data}"
    response = client.chat.completions.create(
        model = 'deepseek-chat',
        messages =[
            {"role":"system","content":"You are a helpful AI assistant for entity extraction."},
            {"role":"user","content":prompt},
        ],
        stream=False,
    )
    res = response.choices[0].message.content
    return res