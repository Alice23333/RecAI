# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# two optional methods for generation
#   1. one-step: generating user query and plans at the same time
# √ 2. two-step: generating user query first, and then come up the plans. But the way is not friendly to multi-turn conversation
# two-step method is inspired by SELF-INSTRUCT (http://arxiv.org/abs/2212.10560)

# TODO: add multi-turn conversations

############ 我的配置 #############
import os
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

from dotenv import load_dotenv
load_dotenv(dotenv_path="D:/RecAI/RecAI-main/InteRecAgent/.env", override=True)
##################################


import os
import re
import sys
import copy
import time
import json
import random
import pickle
import argparse
# import guidance
# from guidance import models, gen, system, user, assistant
from typing import *
from openai import OpenAI

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))

from llm4crs.prompt import *
from llm4crs.utils import replace_substrings

parser = argparse.ArgumentParser(prog="Demostration Generator")
parser.add_argument('--domain', type=str, default='game')
# parser.add_argument("-e", "--engine", type=str, help="deployed LLM engine name, dependend on the deployment")
parser.add_argument("-n", "--num", type=int, default=10, help="number of demostrations to be generated in input-first mode; number of demonstrations for each tool using plan in output-first mode")
# parser.add_argument("-m", "--model", type=str, choices=['gpt-4', 'gpt-3.5-turbo', 'text-davinci-003'], default="text-davinci-003", help="LLM model name to use, must be the name given by OpenAI")
parser.add_argument("-m", "--model", type=str, choices=['gpt-4', 'gpt-3.5-turbo', 'text-davinci-003'], default="gpt-3.5-turbo", help="LLM model name to use, must be the name given by OpenAI")
parser.add_argument("-d", "--dir", type=str, default="./gen_demos", help="directory path to save the generated demonstrations")
parser.add_argument("-s", "--seed", type=str, default="./seed_demos.jsonl", help="file path of seed demos")
parser.add_argument("-ns", "--num_seed", type=int, default=0, help="number of seed demos put into generating prompt. If not positive, use all seed demos.")
parser.add_argument("-md", "--mode", type=str, choices=['input-first', 'output-first'], default='input-first', help="If input-first, user request would be generated first, and then plans are given. \
                    In output-first mode, LLM would generate user-request according to given plans and then generate plans for the request, only those whose generated plans are consistent with given plans would be kept.")
parser.add_argument("--check_consistency", action="store_true", help="whether to check the consistency of given plan and generated plan in output-first mode and only save the consistent cases.")
parser.add_argument("--verbose", action="store_true", help="whether to print details in generation.")
args, _ = parser.parse_known_args()



def read_jsonl(fpath: str) -> List[Dict]:
    res = []
    with open(fpath, 'r') as f:
        for line in f:
            data = json.loads(line)
            res.append(data)
    return res


def sample_seed_demo(demos: List[Dict], n: int=-1) -> List[Dict]:
    if n <= 0:
        return demos
    elif n >= len(demos):
        print("WARNING: there is only {} seed demos, while require {}, all seed demos are used.".format(len(demos), n))
    else:
        _demo = copy.deepcopy(demos)
        random.shuffle(_demo)
        return _demo[:n]


def process_generated_requests(requests: str) -> List[str]:
    return re.findall(r'\d+\.\s(.+)', requests)

def process_generated_requests2(requests: str) -> List[str]:
    return [
        line.lstrip("- ").strip()
        for line in requests.splitlines()
        if line.strip()
        ]


def write_jsonl(obj: List[Dict], fpath: str) -> None:
    try:
        with open(fpath, 'w') as outfile:
            for entry in obj:
                json.dump(entry, outfile)
                outfile.write('\n')
        print("Sucessfully saved into {}.".format(fpath))
    except Exception as e:
        print(f"Error {e} raised. The temp file would be saved in {fpath}.pkl")
        with open(f"{fpath}.pkl", 'wb') as tempfile:
            pickle.dump(obj, tempfile)
    return

domain_map = {'item': args.domain, 'Item': args.domain.capitalize(), 'ITEM': args.domain.upper()}
tool_names = {k: v.format(**domain_map) for k,v in TOOL_NAMES.items()}
available_tools = '[' + ','.join(tool_names.keys()) + ']'
tools_desc = OVERALL_TOOL_DESC.format(**domain_map)

# guidance.llm = guidance.llms.OpenAI(
#     model=args.model,
#     api_type=os.environ.get("OPENAI_API_TYPE", 'open_ai'),
#     api_version=os.environ.get("OPENAI_API_VERSION", "2022-12-01" if '4' not in args.model else "2023-03-15-preview"),
#     api_key=os.environ.get("OPENAI_API_KEY", ""),
#     api_base=os.environ.get("OPENAI_API_BASE", None),
#     deployment_id=args.engine,
#     chat_mode='auto',
#     max_retries=10
# )

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_API_BASE", None)
)


# gen_request_directly = guidance('''{{#system~}}
# You are a helpful assistant.
# {{~/system}}
# {{#user~}}
# Assume that you are a user on {{item}} platform, you are looking from some {{item}}s, and you would ask a conversational recommendation system for help. You would give the request. 
# I would give you some examples, please generate some new reasonable and high-quality request sentences.
# {{~/user}}
# {{#assistant~}}
# Ok, I will do that.
# {{~/assistant}}
# {{#user~}}
# Here are some examples of user request:
# {{~! display the few-shot examples ~}}
# {{~#each examples}}
# - {{this.request}}
# {{~/each}}

# Never use specific {{item}} names or {{item}} types. Instead, use placeholders. For example, {{ITEM}} for names, TYPE for types, PRICE for price, DATE for date. 
# The focus is on generating sentence patterns for questions. 
# Now, it's your turn. Please generate {{number}} new request sentences.
# {{~/user}}
# {{#assistant~}}
# {{gen 'requests' temperature=1.0 max_tokens=1000}}
# {{~/assistant}}
# ''')

def gen_request_directly2(number, examples, domain_map, client, model="gpt-4o-mini"):
    example_text = "\n".join([f"- {e['request']}" for e in examples])

    prompt = f"""
You are a helpful assistant.

Assume that you are a user on {domain_map['item']} platform, you are looking from some {domain_map['item']}s, and you would ask a conversational recommendation system for help. You would give the request. 
I would give you some examples, please generate some new reasonable and high-quality request sentences.

Here are some examples of user request:
{example_text}

Never use specific {domain_map['item']} names or {domain_map['item']} types. Instead, use placeholders. For example, {domain_map['ITEM']} for names, TYPE for types, PRICE for price, DATE for date. 
The focus is on generating sentence patterns for questions. 
Now, generate {number} new request sentences.
"""
    logging.info("Generating requests...")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=1.0,
        max_tokens=1000
    )
    content = response.choices[0].message.content.strip()
    logging.info(f"Successfully generated requests: {content}")
    return content

                             

# gen_request_with_plan = guidance('''{{#system~}}
# You are a helpful assistant and good planner.
# {{~/system}}
# {{#user~}}
# In a conversational recommendation system, user would give some requests for {{item}} recommendations. 

# Human requests typically fall under chit-chat, {{item}} info, or {{item}} recommendations. There are some tools to use to deal with human request.\
# For chit-chat, respond with your knowledge. For {{item}} info, use the {{LookUpTool}}. \
# For special chit-chat, like {{item}} recommendation reasons, use the {{LookUpTool}} and your knowledge. \
# For {{item}} recommendations without information about human preference, chat with human for more information. \
# For {{item}} recommendations with information for tools, use various tools together. \

# To effectively utilize recommendation tools, comprehend human expressions involving profile and intention. \
# Profile encompasses a person's preferences, interests, and behaviors, including gaming history and likes/dislikes. \
# Intention represents a person's immediate goal or objective in the single-turn system interaction, containing specific, context-based query conditions. \

# Human intentions consist of hard and soft conditions. \
# Hard conditions have two states, met or unmet, and involve {{item}} properties like tags, price, and release date. \
# Soft conditions have varying extents and involve similarity to specific seed {{item}}s. Separate hard and soft conditions in requests. \

# Here are the tools could be used: 

# {{tools_desc}}

# All SQL commands are used to search in the {{item}} information table (a sqlite3 table).

# If human is looking up information of {{item}}s, such as the description of {{item}}s, number of {{item}}s, price of {{item}}s and so on, use the {{LookUpTool}}. \

# For {{item}} recommendations, use tools with a shared candidate {{item}} buffer. Buffer is initialized with all {{item}}s. Filtering tools fetch candidates from the buffer and update it. \
# Ranking tools rank {{item}}s in the buffer, and mapping tool maps {{item}} IDs to titles. \
# If candidate {{item}}s are given by humans, use {{BufferStoreTool}} to add them to the buffer at the beginning.

# Think about whether to use tool first. If yes, make tool using plan. \
# Only those tool names are optional when making plans: {{tool_names}}

# Your task is to generate user request with a given plan. 
# Never use specific {{item}} names or {{item}} types. Instead, use placeholders. For example, {{ITEM}} for names, TYPE for types, PRICE for price, DATE for date. 
# The focus is on generating sentence patterns for questions. 
# Now, it's your turn. Please generate {{number}} new request sentences.
# {{~/user}}
# {{#assistant~}}
# Ok, I could do it. Could you give me some examples?
# {{~/assistant}}
# {{#user~}}
# Here are some examples of human request and corresponding tool using plan:
# ----
# {{~! display the few-shot examples ~}}
# {{~#each examples}}
# Plan: {{this.plan}}
# Request: {{this.request}}
# ----
# {{~/each}}

# Now, it's your turn. Please generate {{number}} possible user requests to the tool using plan, each request per line. 

# Plan: {{plan}}

# Request 1: xxxx
# ...
# Request {{number}}: xxxx
# {{~/user}}
# {{#assistant~}}
# {{gen 'requests' temperature=1.0 max_tokens=1000}}
# {{~/assistant}}
# ''')

def gen_request_with_plan2(tools_desc, tool_names: Dict, number, examples, plan, domain_map, client, model="gpt-4o-mini"):
    tool_name_text = ", ".join(tool_names.values())

    example_text = "\n----\n".join(
        [f"Plan: {e['plan']}\nRequest: {e['request']}" for e in examples]
    )

    prompt = f"""
You are a helpful assistant and good planner.

In a conversational recommendation system, user would give some requests for {domain_map['item']} recommendations. 

Human requests typically fall under chit-chat, {domain_map['item']} info, or {domain_map['item']} recommendations. There are some tools to use to deal with human request.
For chit-chat, respond with your knowledge. For {domain_map['item']} info, use the {tool_names['LookUpTool']}. 
For special chit-chat, like {domain_map['item']} recommendation reasons, use the {tool_names['LookUpTool']} and your knowledge. 
For {domain_map['item']} recommendations without information about human preference, chat with human for more information. 
For {domain_map['item']} recommendations with information for tools, use various tools together. 

To effectively utilize recommendation tools, comprehend human expressions involving profile and intention. 
Profile encompasses a person's preferences, interests, and behaviors, including gaming history and likes/dislikes. 
Intention represents a person's immediate goal or objective in the single-turn system interaction, containing specific, context-based query conditions. 

Human intentions consist of hard and soft conditions. 
Hard conditions have two states, met or unmet, and involve {domain_map['item']} properties like tags, price, and release date. 
Soft conditions have varying extents and involve similarity to specific seed {domain_map['item']}s. Separate hard and soft conditions in requests. 

Here are the tools could be used: 

{tools_desc}

All SQL commands are used to search in the {domain_map['item']} information table (a sqlite3 table).

If human is looking up information of {domain_map['item']}s, such as the description of {domain_map['item']}s, number of {domain_map['item']}s, price of {domain_map['item']}s and so on, use the {tool_names['LookUpTool']}. 

For {domain_map['item']} recommendations, use tools with a shared candidate {domain_map['item']} buffer. Buffer is initialized with all {domain_map['item']}s. Filtering tools fetch candidates from the buffer and update it. 
Ranking tools rank {domain_map['item']}s in the buffer, and mapping tool maps {domain_map['item']} IDs to titles. 
If candidate {domain_map['item']}s are given by humans, use {tool_names['BufferStoreTool']} to add them to the buffer at the beginning.

Think about whether to use tool first. If yes, make tool using plan. 
Only those tool names are optional when making plans: {tool_name_text}

Your task is to generate user request with a given plan. 
Never use specific {domain_map['item']} names or {domain_map['item']} types. Instead, use placeholders. For example, {domain_map['ITEM']} for names, TYPE for types, PRICE for price, DATE for date. 
The focus is on generating sentence patterns for questions. 
Now, it's your turn. Please generate {number} new request sentences.

Here are some examples of human request and corresponding tool using plan:
----
{example_text}
----

Now generate {number} possible user requests to the tool using plan, each request per line. 

Plan: {plan}

Format:
Request 1: ...
Request 2: ...
...
Request {number}: ...
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant and good planner."},
            {"role": "user", "content": prompt}
        ],
        temperature=1.0,
        max_tokens=1000
    )

    return response.choices[0].message.content.strip()




#============== Step2: generating plans ==============

# make_plan = guidance('''{{#system~}}
# You are a helpful assistant and good planner.
# {{~/system}}
# {{#user~}}
# Your task is to make tool using plans to help human find {{item}}s they are interested in. \

# Human requests typically fall under chit-chat, {{item}} info, or {{item}} recommendations. There are some tools to use to deal with human request.\
# For chit-chat, respond with your knowledge. For {{item}} info, use the {{LookUpTool}}. \
# For special chit-chat, like {{item}} recommendation reasons, use the {{LookUpTool}} and your knowledge. \
# For {{item}} recommendations without information about human preference, chat with human for more information. \
# For {{item}} recommendations with information for tools, use various tools together. \

# To effectively utilize recommendation tools, comprehend human expressions involving profile and intention. \
# Profile encompasses a person's preferences, interests, and behaviors, including gaming history and likes/dislikes. \
# Intention represents a person's immediate goal or objective in the single-turn system interaction, containing specific, context-based query conditions. \

# Human intentions consist of hard and soft conditions. \
# Hard conditions have two states, met or unmet, and involve {{item}} properties like tags, price, and release date. \
# Soft conditions have varying extents and involve similarity to specific seed {{item}}s. Separate hard and soft conditions in requests. \

# Here are the tools could be used: 

# {{tools_desc}}

# All SQL commands are used to search in the {{item}} information table (a sqlite3 table).

# If human is looking up information of {{item}}s, such as the description of {{item}}s, number of {{item}}s, price of {{item}}s and so on, use the {{LookUpTool}}. \

# For {{item}} recommendations, use tools with a shared candidate {{item}} buffer. Buffer is initialized with all {{item}}s. Filtering tools fetch candidates from the buffer and update it. \
# Ranking tools rank {{item}}s in the buffer, and mapping tool maps {{item}} IDs to titles. \
# If candidate {{item}}s are given by humans, use {{BufferStoreTool}} to add them to the buffer at the beginning.

# Think about whether to use tool first. If yes, make tool using plan. \
# Only those tool names are optional when making plans: {{tool_names}}

# Assume that you play a role of tool using planner, I would give you a user request, and you should help me to make the tool using plan.
# {{~/user}}
# {{#assistant~}}
# Ok, I could do it. Could you give me some examples?
# {{~/assistant}}
# {{#user~}}
# Here are some examples of human request and corresponding tool using plan:
# ----
# {{~! display the few-shot examples ~}}
# {{~#each examples}}
# Request: {{this.request}}
# Plan: {{this.plan}}
# ----
# {{~/each}}

# Now, it's your turn. Please make the tool using plan of below requests. 

# Request: {{request}}
# Plan: 
# {{~/user}}
# {{#assistant~}}
# {{gen 'plan' temperature=1.0 max_tokens=400}}
# {{~/assistant}}
# ''')

def make_plan2(tools_desc, tool_names, request, examples, domain, client, model="gpt-4o-mini"):
    print(f'tool_names:{type(tool_names)}, {tool_names}')
    tool_name_text = ", ".join(list(tool_names.values()))
    # print(f'tool_name_text:{type(tool_name_text)}, {tool_name_text}')
    example_text = "\n----\n".join(
        [f"Request: {e['request']}\nPlan: {e['plan']}" for e in examples]
    )

    prompt = f"""
You are a helpful assistant and good planner.

Your task is to make tool using plans to help human find {domain}s they are interested in. 

Human requests typically fall under chit-chat, {domain} info, or {domain} recommendations. There are some tools to use to deal with human request.
For chit-chat, respond with your knowledge. For {domain} info, use the {tool_names['LookUpTool']}. 
For special chit-chat, like {domain} recommendation reasons, use the {tool_names['LookUpTool']} and your knowledge. 
For {domain} recommendations without information about human preference, chat with human for more information. 
For {domain} recommendations with information for tools, use various tools together. 

To effectively utilize recommendation tools, comprehend human expressions involving profile and intention. 
Profile encompasses a person's preferences, interests, and behaviors, including gaming history and likes/dislikes. 
Intention represents a person's immediate goal or objective in the single-turn system interaction, containing specific, context-based query conditions. 

Human intentions consist of hard and soft conditions. 
Hard conditions have two states, met or unmet, and involve {domain} properties like tags, price, and release date. 
Soft conditions have varying extents and involve similarity to specific seed {domain}s. Separate hard and soft conditions in requests. 

Here are the tools could be used: 

{tools_desc}

All SQL commands are used to search in the {domain} information table (a sqlite3 table).

If human is looking up information of {domain}s, such as the description of {domain}s, number of {domain}s, price of {domain}s and so on, use the {tool_names['LookUpTool']}. 

For {domain} recommendations, use tools with a shared candidate {domain} buffer. Buffer is initialized with all {domain}s. Filtering tools fetch candidates from the buffer and update it. 
Ranking tools rank {domain}s in the buffer, and mapping tool maps {domain} IDs to titles. 
If candidate {domain}s are given by humans, use {tool_names['BufferStoreTool']} to add them to the buffer at the beginning.

Think about whether to use tool first. If yes, make tool using plan. 
Only those tool names are optional when making plans: {tool_name_text}

Assume that you play a role of tool using planner, I would give you a user request, and you should help me to make the tool using plan.

Here are some examples of human request and corresponding tool using plan:
{example_text}

Now make the tool using plan of below requests. 

Request: {request}
Plan:
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a good planner."},
            {"role": "user", "content": prompt}
        ],
        temperature=1.0,
        max_tokens=400
    )
    content = response.choices[0].message.content.strip()
    return content



def gen_examples_input_first(client, model: str, n: int, seed_demos: List[Dict], verbose: bool=False) -> List[Dict]:
    for _ in range(3):
        try:
            # out = gen_request_directly(number=n, examples=seed_demos, **domain_map)
            # user_request = process_generated_requests(out['requests'])

            raw_output = gen_request_directly2(number=n, examples=seed_demos, domain_map=domain_map, client=client, model=model)
            # print(f'raw_output:\n{raw_output}')
            user_request = process_generated_requests2(raw_output)
            print(f'user_request:\n{user_request}')

            res = [None] * len(user_request)
            logging.info(f"Successfully generated {len(user_request)} requests.")
            break
        except:
            raise Exception("Something went wrong in generating requests. Please retry.")
    for i, r in enumerate(user_request):
        time.sleep(random.randint(0, 10))
        try:
            # _res = make_plan(tools_desc=tools_desc, tool_names=available_tools, request=r, 
            #                  examples=sample_seed_demo(seed_demos, n=args.num_seed), **domain_map, **tool_names)
            # res[i] = {"request": r, "plan": _res['plan']}

            _res = make_plan2(tools_desc=tools_desc, tool_names=tool_names, request=r, 
                             examples=sample_seed_demo(seed_demos, n=args.num_seed), domain=domain_map['item'], client=client, model=model)
            res[i] = {"request": r, "plan": _res}
            print(f'res[{i}]: {res[i]}')
            if verbose:
                # print("Request-{}: {}\nPlan: {}\n".format(i, r, _res['plan']))
                print("Request-{}: {}\nPlan: {}\n".format(i, r, _res))
        except Exception as e:
            print("Error occurs when processing request-{}. \n    Error: {}".format(r, e))
            continue
    return res


def gen_examples_output_first(client, model: str, n_per_plan: int, seed_demos: List[Dict], possible_plans: List[str], check_consistency: bool=False, verbose: bool=False) -> List[Dict]:
    def check_plan_consistency(input: str, target: str):
        target = re.findall(r'\d+\.\s+(.*?)(?=;\s+\d+\.|\Z)', target)
        pattern = r'\b' + r'\b.*?\b'.join(re.escape(word) for word in target) + r'\b'
        match = re.search(pattern, input)
        return match is not None

    res = []
    for plan in possible_plans:
        try:
            # out = gen_request_with_plan(tools_desc=tools_desc, tool_names=available_tools, number=n_per_plan, 
            #                             examples=sample_seed_demo(seed_demos, n=args.num_seed), plan=plan, **domain_map, **tool_names)
            out = gen_request_with_plan2(tools_desc=tools_desc, tool_names=tool_names, number=n_per_plan, 
                                        examples=sample_seed_demo(seed_demos, n=args.num_seed), plan=plan, domain_map=domain_map, client=client, model=model)
            requests = re.findall(r'\d+\:\s(.+)', out['requests'])
        except:
            continue
        for r in requests:
            try:
                # _plan = make_plan(tools_desc=tools_desc, tool_names=available_tools, request=r, 
                #              examples=sample_seed_demo(seed_demos, n=args.num_seed), **domain_map, **tool_names)
                _plan = make_plan2(tools_desc=tools_desc, tool_names=tool_names, request=r, 
                             examples=sample_seed_demo(seed_demos, n=args.num_seed), domain=domain_map['item'])
                if verbose:
                    # print("Request: {}\nAim Plan: {}\nGen Plan: {}".format(r, plan, _plan['plan']))
                    print("Request: {}\nAim Plan: {}\nGen Plan: {}".format(r, plan, _plan))
                if check_consistency:
                    # if check_plan_consistency(_plan['plan'], plan):
                    #     res.append({'request': r, 'plan': _plan['plan']})
                    if check_plan_consistency(_plan, plan):
                        res.append({'request': r, 'plan': _plan})
                        if verbose:
                            print("Consistency: Ture")
                    else:
                        if verbose:
                            print("Consistency: False")
                        pass
                else:
                    # res.append({'request': r, 'plan': _plan['plan']})
                    res.append({'request': r, 'plan': _plan})

            except Exception as e:
                print("Error occurs when processing request-{}. \n    Error: {}".format(r, e))
                continue

            if verbose:
                print('-----')
            
    return res




#============== Step0: load seed demos ==============
seed_demos = []
seeds = read_jsonl(args.seed)
for case in seeds:
    seed_demos.append({'request': case['request'], 'plan': replace_substrings(case['plan'], tool_names)})
logging.info(f'Successfully loaded {len(seed_demos)} seed_demos.')


#============== Step1: generating demos ==============
# Now LLM generates plan for only one request per time. Maybe it's a little bit expensive if using OpenAI API.
if args.mode == 'input-first':
    res = gen_examples_input_first(client=client, model=args.model, n=args.num, seed_demos=seed_demos, verbose=args.verbose)

elif args.mode == 'output-first':
    # generate all possible plans
    all_tools = list(tool_names.values())

    possible_plans = ["Don't use tool", f"1. {tool_names['LookUpTool']}"]

    def get_possible_plans(tools: List) -> List[str]:
        sub_sets = [[]]
        for x in tools[:-2]:
            sub_sets.extend([item + [x] for item in sub_sets])

        res = []
        for s in sub_sets:
            if len(s) > 0:
                res.append("".join([f"{i+1}. {t}; " for i, t in enumerate(s)]) + f"{len(s)+1}. {tools[-2]}; " + f"{len(s)+2}. {tools[-1]}")
        return res

    possible_plans += get_possible_plans(all_tools)
    res = gen_examples_output_first(client=client, model=args.model, n_per_plan=args.num, seed_demos=seed_demos, possible_plans=possible_plans, check_consistency=args.check_consistency, verbose=args.verbose)


#============== Step3: Save examples ==============
if not os.path.exists(args.dir):
    os.makedirs(args.dir)

now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
fname = f"{now}_{args.mode}.jsonl"

if len(res) > 0:
    write_jsonl(res, os.path.join(args.dir, fname))

print("End")