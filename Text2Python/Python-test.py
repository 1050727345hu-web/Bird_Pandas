#!/usr/bin/env python3
import argparse
import json
import os
import sqlite3
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
import time 
from datetime import datetime, timedelta 
import re 
import sys 
import openai 

def is_network_error(exception):
    basic_network_errors = (
        ConnectionError,
        TimeoutError,
        OSError, 
    )
    
    if isinstance(exception, basic_network_errors):
        return True
    
    openai_network_errors = []
    
    if hasattr(openai, 'APIConnectionError'):
        openai_network_errors.append(openai.APIConnectionError)
    if hasattr(openai, 'APITimeoutError'):
        openai_network_errors.append(openai.APITimeoutError)
    if hasattr(openai, 'RateLimitError'):
        openai_network_errors.append(openai.RateLimitError)
    if hasattr(openai, 'InternalServerError'):
        openai_network_errors.append(openai.InternalServerError)
    if hasattr(openai, 'BadGatewayError'):
        openai_network_errors.append(openai.BadGatewayError)
    if hasattr(openai, 'ServiceUnavailableError'):
        openai_network_errors.append(openai.ServiceUnavailableError)
    
    if openai_network_errors and isinstance(exception, tuple(openai_network_errors)):
        return True
    
    error_msg = str(exception).lower()
    network_keywords = [
        'connection', 'timeout', 'network', 'socket', 'dns', 
        'unreachable', 'refused', 'reset', 'broken pipe',
        'ssl', 'certificate', 'handshake', 'read timeout',
        'connection reset', 'connection aborted', 'tls',
        'eof', 'connection error', 'connect error'
    ]
    
    return any(keyword in error_msg for keyword in network_keywords)

def new_directory(path):  
    if not os.path.exists(path):  
        os.makedirs(path)  

def load_checkpoint_map(checkpoint_path):
    processed = {}
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        qid = str(obj.get('question_id'))
                        if qid:
                            processed[qid] = obj
                    except Exception:
                        continue
        except Exception:
            pass
    return processed

def append_checkpoint(checkpoint_path, entry):
    if not checkpoint_path:
        return
    try:
        line = json.dumps(entry, ensure_ascii=False)
        with open(checkpoint_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
        with open(checkpoint_path + '.bak', 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        pass

def generate_schema_prompt(db_path, num_rows=None):
    full_schema_prompt_list = []
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    schemas = {}
    
    for table in tables:
        if table[0] == 'sqlite_sequence':
            continue
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='{}';".format(table[0]))
        create_prompt = cursor.fetchone()[0]
        schemas[table[0]] = create_prompt
        
        if num_rows:
            cur_table = f"`{table[0]}`" if table[0] in ['order', 'by', 'group'] else table[0]
            cursor.execute(f"SELECT * FROM {cur_table} LIMIT {num_rows}")
            column_names = [description[0] for description in cursor.description]
            values = cursor.fetchall()
            rows_prompt = nice_look_table(column_names=column_names, values=values)
            verbose_prompt = f"/* \n {num_rows} example rows: \n SELECT * FROM {cur_table} LIMIT {num_rows}; \n {rows_prompt} \n */"
            schemas[table[0]] = f"{create_prompt} \n {verbose_prompt}"

    return "\n\n".join(schemas.values())

def nice_look_table(column_names: list, values: list):
    rows = []
    widths = [max(len(str(value[i])) for value in values + [column_names]) for i in range(len(column_names))]
    header = ''.join(f'{column.rjust(width)} ' for column, width in zip(column_names, widths))
    for value in values:
        row = ''.join(f'{str(v).rjust(width)} ' for v, width in zip(value, widths))
        rows.append(row)
    return header + '\n' + "\n".join(rows)

def generate_golden_standard_prompt(question, reference_sql, knowledge=None):
    lines = []

    lines.append('# Task Description:')
    lines.append('# Generate runnable pandas code only. No explanations, no markdown, no JSON.')
    lines.append('# Requirements:')
    lines.append("# 1) Use: import pandas as pd")
    lines.append("# 2) Read tables strictly via pd.read_csv('<table>.csv')")
    lines.append("# 3) Do NOT use or mention SQL/SELECT/JOIN/CREATE/WHERE/etc.")
    lines.append("# 4) Do NOT define functions or classes (no 'def', 'lambda', 'class')")
    lines.append("# 5) Prefer clear variable names; keep code executable end-to-end")
    lines.append("# 6) Use result to record the final result, and finally print(result) to print the final result.")

    if knowledge:
        lines.append(f"# External Knowledge: {knowledge}")

    lines.append(f"# Question: {question}")

    return "\n".join(lines)

def generate_combined_prompts_one(db_path, question, sql, knowledge=None):
    schema_prompt = generate_schema_prompt(db_path)
    comment_prompt = generate_golden_standard_prompt(question, sql, knowledge)
    return schema_prompt + '\n\n' + comment_prompt + '\nCODE '

def init_model(model_name):
    print(f"Loading model from: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def get_model_response(client, prompt, engine):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert Python code generator specializing in pandas data analysis. "
                "Return runnable Python code only. No explanations or markdown. "
                "Strictly use 'import pandas as pd' and pd.read_csv('<table>.csv'). "
                "Do NOT use or mention SQL/SELECT/JOIN/CREATE/WHERE/etc. "
                "Do NOT define functions or classes (no 'def', 'lambda', 'class')."
            )
        },
        {"role": "user", "content": prompt}
    ]

    response_stream = client.chat.completions.create(
        model=f"{engine}", 
        messages=messages,
        max_tokens=8000,
        stream=True,
        extra_body={"enable_thinking": True}
    )

    full_content = ""
    reasoning_content = ""
    for chunk in response_stream:
        if chunk.choices:
            delta = chunk.choices[0].delta
            current_content = getattr(delta, 'content', '') or ''
            full_content += str(current_content)
            current_reasoning = getattr(delta, 'reasoning_content', '') or ''
            reasoning_content += str(current_reasoning)

    content = full_content.strip()

    if "</think>" in content:
        content = content.split("</think>")[-1].strip()

    code = content
    lowered = content.lower()
    if "```python" in lowered:
        try:
            code = content.split("```python", 1)[1].split("```", 1)[0].strip()
        except Exception:
            code = content
    elif "```" in content:
        try:
            code = content.split("```", 1)[1].split("```", 1)[0].strip()
        except Exception:
            code = content

    return {
        "code": code.strip(),
        "thinking": reasoning_content.strip()
    }

def collect_response_from_gpt(db_path_list, question_list, question_id_list, api_key, sql_list, knowledge_list=None, checkpoint_path=None):
    if checkpoint_path:
        new_directory(os.path.dirname(checkpoint_path) or '.')
    checkpoint_map = load_checkpoint_map(checkpoint_path)
    if checkpoint_map:
        print(f"A checkpoint has been found, {len(checkpoint_map)} history has been loaded, and completed entries will be skipped")

    response_list = []
    client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    prompt_dir = "./logs/prompts"
    new_directory(prompt_dir)

    total_questions = len(question_list)
    start_time = time.time()

    for i, question in enumerate(question_list):
        question_id = question_id_list[i]
        qid_key = str(question_id)
        if qid_key in checkpoint_map:
            response_list.append(checkpoint_map[qid_key])
            continue

        elapsed_time = time.time() - start_time
        avg_time_per_question = elapsed_time / (i + 1)
        remaining_time = avg_time_per_question * (total_questions - i - 1)
        task_duration_str = str(timedelta(seconds=int(elapsed_time)))
        remaining_str = str(timedelta(seconds=int(remaining_time)))
        completion_time = (datetime.now() + timedelta(seconds=remaining_time)).strftime("%H:%M:%S")

        print(f"\n[Progress: {i+1}/{total_questions}]")
        print(f"✓ Time:{task_duration_str} | Remaining:{remaining_str} | Estimated:{completion_time} | Progress:{((i+1)/total_questions*100):.1f}%")

        try:
            cur_prompt = generate_combined_prompts_one(
                db_path=db_path_list[i],
                question=question,
                sql=sql_list[i],
                knowledge=knowledge_list[i] if knowledge_list else None
            )

            try:
                with open(os.path.join(prompt_dir, f"{question_id}.txt"), 'w', encoding='utf-8') as f:
                    f.write(cur_prompt)
            except Exception:
                pass

            resp = get_model_response(client, cur_prompt, args.engine)
            code = resp.get('code', '').strip()
            thinking = resp.get('thinking', '').strip()

            db_id = db_path_list[i].split('/')[-1].split('.sqlite')[0]
            response_entry = {
                'question': question,
                'code': code,
                'thinking': thinking,
                'db_id': db_id,
                'question_id': question_id
            }
            response_list.append(response_entry)
            append_checkpoint(checkpoint_path, response_entry)

        except Exception as e:
            print(f"Error processing question {question_id}: {str(e)}")
            if is_network_error(e):
                print("❌ Network error detected, program stopped")
                print(f"Error details: {str(e)}")
                print(f"Processed {len(response_list)} questions")
                sys.exit(1)

            db_id = db_path_list[i].split('/')[-1].split('.sqlite')[0]
            response_entry = {
                'question': question,
                'code': '',
                'thinking': f'Processing failed: {str(e)}',
                'db_id': db_id,
                'question_id': question_id,
                'error': str(e)
            }
            response_list.append(response_entry)
            append_checkpoint(checkpoint_path, response_entry)

    return response_list

def decouple_question_schema(datasets, db_root_path):
    question_id_list = []
    question_list = []
    db_path_list = []
    knowledge_list = []
    sql_list = []
    for data in datasets:
        question_id_list.append(data['question_id'])
        question_list.append(data['question'])
        db_path_list.append(f"{db_root_path}{data['db_id']}/{data['db_id']}.sqlite")
        
        evidence = data.get('evidence', '')
        verified_knowledge = data.get('verified_knowledge', '')
        
        combined_knowledge = []
        if evidence:
            combined_knowledge.append(f"Evidence: {evidence}")
        if verified_knowledge:
            combined_knowledge.append(f"Verified Knowledge: {verified_knowledge}")
        
        knowledge_list.append(" | ".join(combined_knowledge) if combined_knowledge else "")
        sql_list.append(data['sql'])
    return question_id_list, question_list, db_path_list, knowledge_list, sql_list

def generate_code_file(code_list, output_path=None):
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(code_list, f, indent=4, ensure_ascii=False)
    return code_list

if __name__ == '__main__':
    args = argparse.Namespace(
        eval_path='../Verified_Bird_Python.json',
        mode='dev',
        use_knowledge='True',
        db_root_path='../dev_databases/',
        api_key="YOUR_API_KEY",
        engine='qwen3-max',  
        data_output_path=None,  
        limit=int(os.environ.get('LIMIT', '0'))  
    )

    if args.data_output_path is None:
        args.data_output_path = f"{args.engine}_output_kg"

    eval_data = json.load(open(args.eval_path, 'r', encoding='utf-8'))
    question_id_list, question_list, db_path_list, knowledge_list, sql_list = decouple_question_schema(
        datasets=eval_data,
        db_root_path=args.db_root_path
    )
    
    if args.limit > 0:
        question_id_list = question_id_list[:args.limit]
        question_list = question_list[:args.limit]
        db_path_list = db_path_list[:args.limit]
        knowledge_list = knowledge_list[:args.limit]
        sql_list = sql_list[:args.limit]

    responses = collect_response_from_gpt(
        db_path_list=db_path_list,
        question_list=question_list,
        question_id_list=question_id_list,
        api_key=args.api_key,
        sql_list=sql_list,
        knowledge_list=knowledge_list if args.use_knowledge == 'True' else None,
        checkpoint_path=os.path.join('logs_max', f'checkpoints_{args.mode}.jsonl')
    )
    output_name = os.path.join(args.data_output_path, f'predict_{args.mode}.json')
    generate_code_file(code_list=responses, output_path=output_name)
    
    print(f'Successfully collected results for {args.mode} dataset')
    print(f'Results saved to: {output_name}')
