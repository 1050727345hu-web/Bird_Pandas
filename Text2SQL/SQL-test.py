#!/usr/bin/env python3
import argparse
import json
import os
import sqlite3
from tqdm import tqdm
from openai import OpenAI
import time
from datetime import datetime, timedelta
import re

def new_directory(path):  
    if not os.path.exists(path):  
        os.makedirs(path)  

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

def generate_comment_prompt(question, knowledge=None):
    pattern = "-- Using valid SQLite, answer the following questions for the tables provided above."
    if knowledge:
        pattern = "-- Using valid SQLite and understanding External Knowledge, answer the following questions for the tables provided above."
        knowledge_prompt = f"-- External Knowledge: {knowledge}"
        return f"{knowledge_prompt}\n{pattern}\n-- {question}"
    return f"{pattern}\n-- {question}"

def generate_combined_prompts_one(db_path, question, knowledge=None):
    schema_prompt = generate_schema_prompt(db_path)
    comment_prompt = generate_comment_prompt(question, knowledge)
    return schema_prompt + '\n\n' + comment_prompt + '\nSELECT '

def extract_sql(response, question_index):
    response_dict = {str(question_index): response}
    
    try:
        with open(f'{engine}_origin_response.json', 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            existing_data.update(response_dict)
    except FileNotFoundError:
        existing_data = response_dict
    
    with open(f'{engine}_origin_response.json', 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)

    if "</think>" in response:
        response = response.split("</think>")[-1].strip()
    
    if "```sql" in response.lower():
        try:
            sql_start = response.lower().find("```sql") + 6
            remaining = response[sql_start:]
            sql_end = remaining.find("```")
            if sql_end != -1:
                sql = remaining[:sql_end].strip()
            else:
                sql = remaining.strip()
        except Exception:
            sql = response.strip()
    elif "```" in response:
        try:
            first_backtick = response.find("```")
            if first_backtick != -1:
                remaining = response[first_backtick + 3:]
                end_backtick = remaining.find("```")
                if end_backtick != -1:
                    sql = remaining[:end_backtick].strip()
                else:
                    sql = remaining.strip()
            else:
                sql = response.strip()
        except Exception:
            sql = response.strip()
    else:
        sql = response.strip()
        
        if "SELECT" in sql.upper():
            select_pos = sql.upper().find("SELECT")
            if select_pos > 0:
                sql = sql[select_pos:]
        
        lines = sql.split('\n')
        sql_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('--') and not line.startswith('#'):
                sql_lines.append(line)
        
        if sql_lines:
            sql = ' '.join(sql_lines)

    sql = sql.strip()
    
    if sql.endswith(';'):
        sql = sql[:-1]
    
    return sql.strip()

def save_prompt_and_sql(question_id, prompt, sql_output):
    logs_dir = f'./logs/{engine}_origin_prompt'
    new_directory(logs_dir)
    
    prompt_file = os.path.join(logs_dir, f'{question_id}_prompt.txt')
    with open(prompt_file, 'w', encoding='utf-8') as f:
        f.write(prompt)
    
    sql_file = os.path.join(logs_dir, f'{question_id}_sql.txt')
    with open(sql_file, 'w', encoding='utf-8') as f:
        f.write(sql_output)

def get_model_response(client, prompt, question_id=None):
    if question_id is not None:
        logs_dir = f'./logs/{engine}_origin_prompt'
        new_directory(logs_dir)
        prompt_file = os.path.join(logs_dir, f'{question_id}_prompt.txt')
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt)
    
    messages = [
        {"role": "system", "content": "You are a SQL assistant. Only return the SQL query without any explanation."},
        {"role": "user", "content": prompt}
    ]
    
    response_stream = client.chat.completions.create(
        model=engine,
        messages=messages,
        max_tokens=8000,
        temperature=0.001,
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
    
    content = full_content
    
    sql = extract_sql(content, question_id if question_id is not None else -1)
    
    if question_id is not None:
        sql_file = os.path.join(logs_dir, f'{question_id}_sql.txt')
        with open(sql_file, 'w', encoding='utf-8') as f:
            f.write(sql.strip())

    return sql.strip()

def collect_response_from_gpt(db_path_list, question_list, api_key, engine, knowledge_list=None):
    response_list = []
    client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    max_questions = 1534
    question_list = question_list[:max_questions]
    db_path_list = db_path_list[:max_questions]
    if knowledge_list:
        knowledge_list = knowledge_list[:max_questions]
    
    logs_dir = f'./logs/{engine}_origin_prompt'
    new_directory(logs_dir)

    checkpoint_file = f'./logs/{engine}_origin_checkpoint.json'
    start_idx = 0
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                response_list = checkpoint_data.get('responses', [])
                start_idx = checkpoint_data.get('next_index', 0)
                print(f"Resuming processing from index {start_idx} (completed {len(response_list)} questions)")
        except Exception as e:
            print(f"Failed to read checkpoint file: {e}")
    
    total_questions = len(question_list)
    start_time = time.time()
    
    for i in range(start_idx, len(question_list)):
        question = question_list[i]
        elapsed_time = time.time() - start_time
        avg_time_per_question = elapsed_time / (i - start_idx + 1) if i > start_idx else 0
        remaining_time = avg_time_per_question * (total_questions - i - 1)
        
        task_duration_str = str(timedelta(seconds=int(elapsed_time)))
        remaining_str = str(timedelta(seconds=int(remaining_time)))
        completion_time = (datetime.now() + timedelta(seconds=remaining_time)).strftime("%H:%M:%S")
        
        print(f"\n[Progress: {i+1}/{total_questions}]")
        print(f"✓ Time:{task_duration_str} | Remaining:{remaining_str} | Estimated:{completion_time} | Progress:{((i+1)/total_questions*100):.1f}%")
        print(f'the question is: {question}')
        
        try:
            cur_prompt = generate_combined_prompts_one(
                db_path=db_path_list[i], 
                question=question,
                knowledge=knowledge_list[i] if knowledge_list else None
            )
            
            sql = get_model_response(client, cur_prompt, question_id=i)
            
            if sql and not sql.upper().startswith('SELECT'):
                sql = 'SELECT ' + sql
                
            db_id = db_path_list[i].split('/')[-1].split('.sqlite')[0]
            sql = sql + '\t----- bird -----\t' + db_id
            response_list.append(sql)
            
            checkpoint_data = {
                'responses': response_list,
                'next_index': i + 1,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error processing question {i}: {str(e)}")
            checkpoint_data = {
                'responses': response_list,
                'next_index': i,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'last_error': str(e)
            }
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False)

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    print(f"Processed {len(response_list)} questions out of {len(question_list)}")
    return response_list

def decouple_question_schema(datasets, db_root_path):
    question_list = []
    db_path_list = []
    knowledge_list = []
    for data in datasets:
        question_list.append(data['question'])
        db_path_list.append(f"{db_root_path}{data['db_id']}/{data['db_id']}.sqlite")
        evidence = data.get('evidence', '')
        verified_knowledge = data.get('verified_knowledge', '')

        if not isinstance(evidence, str):
            evidence = str(evidence) if evidence is not None else ''
        if not isinstance(verified_knowledge, str):
            verified_knowledge = str(verified_knowledge) if verified_knowledge is not None else ''

        combined_knowledge = evidence + verified_knowledge
        knowledge_list.append(combined_knowledge)
    return question_list, db_path_list, knowledge_list

def generate_sql_file(sql_lst, output_path=None):
    result = {i: sql for i, sql in enumerate(sql_lst)}
    if output_path:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4)
            
    return result

if __name__ == '__main__':
    eval_path = '../Origin_dev_Bird.json'
    mode = 'dev'
    use_knowledge = 'True'
    db_root_path = '../dev_databases/'
    api_key = 'YOUR_API_KEY'
    engine = 'qwen3-max'
    data_output_path = f'./exp_result/{engine}_output_kg'
    chain_of_thought = 'True'
    
    eval_data = json.load(open(eval_path, 'r', encoding="utf-8"))
    question_list, db_path_list, knowledge_list = decouple_question_schema(
        datasets=eval_data, 
        db_root_path=db_root_path
    )
    
    responses = collect_response_from_gpt(
        db_path_list=db_path_list,
        question_list=question_list,
        api_key=api_key,
        engine=engine,
        knowledge_list=knowledge_list if use_knowledge == 'True' else None
    )
    
    output_name = os.path.join(data_output_path, f'predict_{mode}.json')
    generate_sql_file(sql_lst=responses, output_path=output_name)
    
    print(f'Successfully collected results from {engine} for {mode} evaluation')
    print(f'Results saved to: {output_name}')
