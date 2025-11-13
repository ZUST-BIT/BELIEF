def get_and_convert_data():
    """ 从mongodb中获取文本数据，并转换成可以embedding的文本 """
    db_name = "bio"
    collection_name = "pubmed"
    url = "mongodb://172.18.51.200:27017/"
    from pymongo import MongoClient
    client = MongoClient(url)
    db = client[db_name]
    collection = db[collection_name]
    res = collection.find()
    texts = []
    metas = []
    for i in res:
        title = i.get('title','') or ''
        abstract = i.get('abstract','') or ''
        # keyword = i.get('keyword',[]) or []  # 后续可以加入关键词部分
        text = title.strip()
        if abstract:
            text += " " + abstract.strip()
        texts.append(text)
        metas.append({
            "id": str(i.get("_id")),
            "title": title,
            "abstract": abstract,
            "keyword": i.get("keyword", []),
            "doi": i.get("doi", ""),
        })
    return texts, metas

# texts,metas = get_and_convert_data()
# print(texts[:5])
# print("--" * 10)
# print(metas[:5])