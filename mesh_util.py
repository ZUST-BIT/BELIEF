# MeSH 标准化工具代码
import json
import os
import time
from config import set_argument

class MeshManager:
    def __init__(self):
        """
        初始化时加载数据库
        """
        # 从 config 获取路径，如果没有配置则使用默认路径
        args = set_argument()
        self.mapping_path = getattr(args, 'mesh_mapping_path', "D:/BitLabData/MeSH/mesh_mapping.json")
        self.info_path = getattr(args, 'mesh_info_path', "D:/BitLabData/MeSH/mesh_info.json")
        
        self.mapping_db = {}
        self.info_db = {}
        self._load_data()

    def _load_data(self):
        """加载 MeSH 数据库到内存"""
        t_start = time.time()
        
        try:
            if not os.path.exists(self.mapping_path) or not os.path.exists(self.info_path):
                print(f"❌ [MeSH] File not found: {self.mapping_path} or {self.info_path}")
                return

            with open(self.mapping_path, 'r', encoding='utf-8') as f:
                self.mapping_db = json.load(f)
                
            with open(self.info_path, 'r', encoding='utf-8') as f:
                self.info_db = json.load(f)
                
            
        except Exception as e:
            print(f"❌ [MeSH] Load failed: {e}")

    def normalize(self, entity_list):
        """
        核心功能：输入实体列表，返回标准化字典
        """
        output_dict = {}
        
        if not self.mapping_db:
            return {e: {"found": False, "mesh_id": None, "standard_name": None, "description": None} for e in entity_list}

        for raw_name in entity_list:
            # 1. 预处理 Key
            search_key = raw_name.lower().strip()
            
            # 2. 初始化结果
            result_obj = {
                "found": False,
                "mesh_id": None,
                "standard_name": raw_name, # 默认用原名，方便后续处理
                "description": ""
            }

            # 3. 查表
            mesh_id = self.mapping_db.get(search_key)
            
            if mesh_id:
                info = self.info_db.get(mesh_id)
                if info:
                    result_obj["found"] = True
                    result_obj["mesh_id"] = mesh_id
                    result_obj["standard_name"] = info['name']
                    # 处理描述
                    desc = info.get('desc')
                    result_obj["description"] = desc if desc else "No description available."

            output_dict[raw_name] = result_obj
            
        return output_dict