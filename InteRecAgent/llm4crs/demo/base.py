# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# 这份代码本质上是一个 Few-shot 示例选择与拼接模块（Demo 选择器），用于给 Agent 动态构造“示例驱动的 Prompt”。
# 它的核心作用是：根据当前用户请求，自动选出合适的示例（demonstrations），并拼成一个 Few-shot Prompt。

############ 我的配置 #############
import os
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(filename)s - %(levelname)s - %(message)s")

# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

# from dotenv import load_dotenv
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
model_path = os.path.join(project_root, 'hf_model', 'all-mpnet-base-v2')
##################################

import os
import json
from langchain.prompts import example_selector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

from typing import *

from llm4crs.utils import replace_substrings_regex

example_prompt = PromptTemplate(
    input_variables=["request", "plan"],
    template="User Request: {request} \nPlan: {plan}",
)

def read_jsonl(fpath: str) -> List[Dict]:
    res = []
    with open(fpath, 'r') as f:
        for line in f:
            data = json.loads(line)
            res.append(data)
    return res


class DemoSelector:

    def __init__(self, mode: str, demo_dir_or_file: str, k: int, domain: str) -> None:
        assert mode in {'fixed', 'dynamic'}, f"Optional demonstration selector mode: 'fixed', 'dynamic', while got {mode}"
        self.mode = mode
        self.k = k
        examples = self.load_examples(demo_dir_or_file)
        self.examples = self.fit_domain(examples, domain)

        input_variables = {
            "example_prompt": example_prompt,
            "example_separator": "\n-----\n",
            "prefix": "Here are some demonstrations of user requests and corresponding tool using plans:",
            "suffix": "Refer to above demonstrations to use tools for current request: {in_request}.", 
            "input_variables": ["in_request"],
        }
        if self.mode == 'dynamic':
            logging.info(f"Loading model from: {model_path}")
            selector = example_selector.SemanticSimilarityExampleSelector.from_examples(
                # This is the list of examples available to select from.
                self.examples, 
                # This is the embedding class used to produce embeddings which are used to measure semantic similarity. 用 HuggingFace 模型生成示例 embedding
                # HuggingFaceEmbeddings(), 
                HuggingFaceEmbeddings(model_name=model_path), 
                # This is the VectorStore class that is used to store the embeddings and do a similarity search over. 存入 Chroma 向量数据库
                Chroma, 
                # This is the number of examples to produce.
                k=self.k
            )
            input_variables["example_selector"] = selector
        else:
            input_variables["examples"] = self.examples[: self.k]
        
        self.prompt = FewShotPromptTemplate(**input_variables)


    def load_examples(self, dir: str) -> List[Dict]:
        examples = []
        if os.path.isdir(dir):
            for f in os.listdir(dir):
                if f.endswith("jsonl"):
                    fname = os.path.join(dir, f)
                    examples.extend(read_jsonl(fname))
        else:
            if dir.endswith('.jsonl'):
                examples.extend(read_jsonl(dir))
        assert len(examples) > 0, "Failed to load examples. Note that only .jsonl file format is supported for demonstration loading. "
        return examples


    def __call__(self, request: str):
        return self.prompt.format(in_request=request)

    # 让通用示例适配不同领域。
    def fit_domain(self, examples: List[Dict], domain: str):
        # fit examples into domains: replace placeholder with domain-related words, like replacing item with movie, game
        domain_map = {'item': domain, 'Item': domain.capitalize(), 'ITEM': domain.upper()}
        
        res = []
        for case in examples:
            if not case:
                continue
            _case = {}
            _case['request'] = replace_substrings_regex(case['request'], domain_map)
            _case['plan'] = replace_substrings_regex(case['plan'], domain_map)
            res.append(_case)
        return res




if __name__ == "__main__":
    # selector = DemoSelector("./LLM4CRS/demonstration/gen_demos/2023-06-28-08_53_56.jsonl", k=3)
    demo_dir_or_file = os.path.join(project_root, 'demonstration', 'gen_demos', '2026-02-15-23_26_59_input-first.jsonl')
    selector = DemoSelector(mode="dynamic", demo_dir_or_file=demo_dir_or_file, k=3, domain="game")
    request = "I want some farming games."

    demo_prompt = selector(request)
    print(demo_prompt)
    print("passed.")