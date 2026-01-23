#!/usr/bin/env python3
import argparse
import json
import os
import sqlite3
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

# ---------------------- Prompt Constructors for 3-Phase Protocol ----------------------

def generate_phase1_prompt(schema_prompt, question, knowledge=None):
    """
    Phase 1: Logic Probing
    Task: Identify ambiguity.
    """
    # Update: Generalized to "Data Analysis" context instead of "Using valid SQLite"
    pattern = "-- As an expert Data Analyst, identify ambiguity in the following question regarding the data schema or business logic."
    base_knowledge = ""
    if knowledge:
        base_knowledge = f"-- External Knowledge: {knowledge}\n"
    
    task_instruction = (
        "\n-- Task: You are a logical analyst. Identify any ambiguity in the question regarding the database schema, business logic, or output structure requirements (e.g., exact columns to return, handling of NULLs).  "
        "Return ONLY a clarifying question that addresses the ambiguity. Do not generate SQL or Code."
    )
    
    return f"{schema_prompt}\n\n{base_knowledge}{pattern}\n-- {question}{task_instruction}"

def generate_phase2_prompt(schema_prompt, question, gold_code, phase1_inquiry, knowledge=None):
    """
    Phase 2: Ground Truth Injection (Oracle)
    Task: Extract the business logic from Gold Code into Natural Language constraints.
    """
    # Update: Based on Ground Truth Code instead of SQL
    pattern = "-- Based strictly on the Ground Truth Code provided below, extract the business logic."
    base_knowledge = ""
    if knowledge:
        base_knowledge = f"-- External Knowledge: {knowledge}\n"

    # Oracle Instruction
    oracle_instruction = f"""
-- Ground Truth Code: {gold_code}
-- Model's Inquiry: {phase1_inquiry}
-- Task: You are a Data Analyst Proxy. Based strictly on the Ground Truth Code provided above, answer the Model's Inquiry by explaining the business constraints or logic used in the Gold Code.

Return your response in the following JSON format:
{{
    "classification": ["Category1", "Category2"],
    "answer": "Your natural language explanation here. Use the BIRD dataset explanation style: concise, direct, avoiding irrelevant sentences. Examples: 'Eligible free rates for students aged 5-17 = `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)`', 'Total enrollment can be represented by `Enrollment (K-12)` + `Enrollment (Ages 5-17)`', 'count of schools; sum of average math scores; \'rtype\' refers to \'S\'', 'Communication number refers to phone number.', 'K-12 refers to students in grades 1 through 12.', 'full name means first name, last name; There are at most 3 administrators for each school; SAT Scores are greater or equal to 1500 refers to NumGE1500', 'year equals 1980 in the database; ratio calculated by dividing 1 by 1; ratio calculated by dividing 12 by 31;', 'Exclusively virtual refers to Virtual = \'F\'; respective counties means PARTITION BY County'"
}}

 -- Classification: Classify the Model's Inquiry into ONE or more of the following categories (use exact category names):
- [Domain Concept Definition]: Questions about what specific terms/concepts map to in the database schema (e.g., "what is a charter school?", "what does 'continuation' mean?")
- [Calculation & Aggregation Logic]: Questions about formulas, ranking rules, or computational methods (e.g., "how is the rate calculated?", "what does 'highest' mean?")
- [Structural & Scope Ambiguity]: Questions about data scope, table relationships, or entity boundaries (e.g., "county vs county office?", "all schools vs specific type?")
- [Constraint & Boundary Specification]: Questions about filters, ranges, null handling, or boundary conditions (e.g., "after 2000?", "inclusive/exclusive?", "null values?")
- [Join & Relationship Logic]: Questions about how tables connect or relate to each other (e.g., "how to match schools to scores?")
- [output structure requirements]:output structure requirements (e.g., exact columns to return, handling of NULLs).
- Output other similar categories by yourself.

-- Constraints:
1.  Do NOT output code.
2.  Extract the business logic from the Code into implementation-agnostic natural language constraints (e.g., "Filter by status 'Active'" instead of "Use df[df['status'] == 'Active']").
3.  Answer ONLY what is explicitly asked in the "Model's Inquiry". Do NOT volunteer extra information about other operations, calculations, or data manipulations that were not queried.
4.  Be CONCISE.  Do not add external background knowledge definitions.
5.  Explicitly mention the specific Column Names and Values required to satisfy the inquiry.
6.  If the inquiry is irrelevant to the code logic, reply 'Standard convention'.
"""
    return f"{schema_prompt}\n\n{base_knowledge}{pattern}\n-- {question}\n{oracle_instruction}"

def generate_phase3_prompt(schema_prompt, question, knowledge, oracle_answer):
    """
    Phase 3: Execution
    Task: Generate pandas code with augmented knowledge.
    """

    augmented_knowledge = str(knowledge) if knowledge else ""
    if oracle_answer:
        augmented_knowledge += f" Note: {oracle_answer}"

    lines = []

    # 任务说明（强调 pandas、CSV、禁止 SQL/函数定义）
    lines.append('# Task Description:')
    lines.append('# Generate runnable pandas code only. No explanations, no markdown, no JSON.')
    lines.append('# Requirements:')
    lines.append("# 1) Use: import pandas as pd")
    lines.append("# 2) Read tables strictly via pd.read_csv('<table>.csv')")
    lines.append("# 3) Do NOT use or mention SQL/SELECT/JOIN/CREATE/WHERE/etc.")
    lines.append("# 4) Do NOT define functions or classes (no 'def', 'lambda', 'class')")
    lines.append("# 5) Prefer clear variable names; keep code executable end-to-end")
    lines.append("# 6) Use result to record the final result, and finally print(result) to print the final result.")

    if augmented_knowledge:
        lines.append(f"# External Knowledge: {augmented_knowledge}")

    lines.append(f"# Question: {question}")

    comment_prompt = "\n".join(lines)

    return f"{schema_prompt}\n\n{comment_prompt}\nCODE "

# --------------------------------------------------------------------------------------

def extract_inquiry_category(phase2_response):
    import json
    import re

    try:
        # 尝试解析JSON格式的回复
        # 首先清理回复，移除可能的markdown代码块标记
        cleaned_response = phase2_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]

        cleaned_response = cleaned_response.strip()

        parsed = json.loads(cleaned_response)

        if isinstance(parsed, dict) and "classification" in parsed:
            categories = parsed["classification"]
            if isinstance(categories, list):
                return categories
            elif isinstance(categories, str):
                return [categories]

    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    category_patterns = [
        r'\[([^\]]+)\]',  
        r'([A-Za-z &]+):\s',  
    ]

    categories = []
    for pattern in category_patterns:
        matches = re.findall(pattern, phase2_response)
        for match in matches:
            clean_match = match.strip()
            if clean_match and len(clean_match) > 3:  
                categories.append(clean_match)

    unique_categories = list(set(categories))
    return unique_categories if unique_categories else ["Unclassified"]

def extract_oracle_answer(phase2_response):
    import json

    try:
        cleaned_response = phase2_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]

        cleaned_response = cleaned_response.strip()
        parsed = json.loads(cleaned_response)

        if isinstance(parsed, dict) and "answer" in parsed:
            return parsed["answer"]

    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    return phase2_response

def extract_code(response, question_index, engine_name):
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()

    code = response
    lowered = response.lower()
    if "```python" in lowered:
        try:
            code = response.split("```python", 1)[1].split("```", 1)[0].strip()
        except Exception:
            code = response
    elif "```" in response:
        try:
            code = response.split("```", 1)[1].split("```", 1)[0].strip()
        except Exception:
            code = response

    return code.strip()

def save_single_phase_log(engine, question_id, phase_name, prompt, response):
    logs_dir = f'./logs/{engine}_origin_prompt'
    new_directory(logs_dir)
    
    prompt_file = os.path.join(logs_dir, f'{question_id}_{phase_name}_prompt.txt')
    with open(prompt_file, 'w', encoding='utf-8') as f:
        f.write(prompt)
        
    resp_file = os.path.join(logs_dir, f'{question_id}_{phase_name}_response.txt')
    with open(resp_file, 'w', encoding='utf-8') as f:
        f.write(response)

def get_model_response(client, prompt, model_name, temperature=0.001, max_tokens=4000, stop=None, is_code_generation=False):
    if is_code_generation:
        system_content = (
            "You are an expert Python code generator specializing in pandas data analysis. "
            "Return runnable Python code only. No explanations or markdown. "
            "Strictly use 'import pandas as pd' and pd.read_csv('<table>.csv'). "
            "Do NOT use or mention SQL/SELECT/JOIN/CREATE/WHERE/etc. "
            "Do NOT define functions or classes (no 'def', 'lambda', 'class')."
        )
    else:
        system_content = "You are a helpful assistant."

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt}
    ]

    if is_code_generation:
        try:
            response_stream = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
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
        except Exception as e:
            print(f"Error calling model {model_name}: {e}")
            return {
                "code": "",
                "thinking": f"Error: {str(e)}"
            }
    else:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling model {model_name}: {e}")
            return str(e)

def collect_response_from_gpt_3phase(db_path_list, question_list, gold_code_list, api_key, engine, knowledge_list=None, limit_count=None):
    
    response_list = [] 
    phase1_results = []
    phase2_results = []

    target_client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    oracle_client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    oracle_model = "qwen3-max" 

    max_questions = limit_count if limit_count is not None else len(question_list)

    question_list = question_list[:max_questions]
    db_path_list = db_path_list[:max_questions]
    gold_code_list = gold_code_list[:max_questions]
    if knowledge_list:
        knowledge_list = knowledge_list[:max_questions]

    logs_dir = f'./logs/{engine}_origin_prompt'
    new_directory(logs_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_pattern = f'./logs/{engine}_origin_checkpoint_*.json'
    import glob
    checkpoint_files = glob.glob(checkpoint_pattern)

    start_idx = 0
    checkpoint_file = None

    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        checkpoint_file = latest_checkpoint
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                response_list = checkpoint_data.get('responses', [])
                phase1_results = checkpoint_data.get('answer_Phase1', [])
                phase2_results = checkpoint_data.get('answer_Phase2', [])
                start_idx = checkpoint_data.get('next_index', 0)
                print(f"Resuming from checkpoint: {checkpoint_file}")
                print(f"Resuming from index {start_idx}")
        except Exception as e:
            print(f"Checkpoint read failed: {e}")
            checkpoint_file = f'./logs/{engine}_origin_checkpoint_{timestamp}.json'
    else:
        checkpoint_file = f'./logs/{engine}_origin_checkpoint_{timestamp}.json'

    total_questions = len(question_list)
    start_time = time.time()

    for i in range(start_idx, total_questions):
        question = question_list[i]
        db_path = db_path_list[i]
        gold_code = gold_code_list[i]
        knowledge = knowledge_list[i] if knowledge_list else None
        
        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / (i - start_idx + 1) if i > start_idx else 0
        remaining = avg_time * (total_questions - i - 1)
        print(f"\n[Progress: {i+1}/{total_questions}]")
        print(f"Remaining: {str(timedelta(seconds=int(remaining)))}")
        print(f'Question: {question}')

        try:
            schema_prompt = generate_schema_prompt(db_path)

            p1_prompt = generate_phase1_prompt(schema_prompt, question, knowledge)
            
            ans_phase1 = get_model_response(target_client, p1_prompt, model_name=engine)
            
            save_single_phase_log(engine, i, "Phase1", p1_prompt, ans_phase1)
            phase1_results.append(ans_phase1)
            print(f"  -> Phase 1 Inquiry: {ans_phase1[:100]}...")

            p2_prompt = generate_phase2_prompt(schema_prompt, question, gold_code, ans_phase1, knowledge)
            
            ans_phase2 = get_model_response(oracle_client, p2_prompt, model_name=oracle_model)
            
            save_single_phase_log(engine, i, "Phase2", p2_prompt, ans_phase2)
            phase2_results.append(ans_phase2)
            print(f"  -> Phase 2 Oracle: {ans_phase2[:100]}...")

            oracle_clean_answer = extract_oracle_answer(ans_phase2)
            p3_prompt = generate_phase3_prompt(schema_prompt, question, knowledge, oracle_answer=oracle_clean_answer)

            resp_p3 = get_model_response(target_client, p3_prompt, model_name=engine, is_code_generation=True)
            code = resp_p3.get('code', '').strip()
            thinking = resp_p3.get('thinking', '').strip()

            db_id = db_path.split('/')[-1].split('.sqlite')[0]
            code_entry = {
                'question': question,
                'code': code,
                'thinking': thinking,
                'db_id': db_id,
                'question_id': question_id_list[i]  
            }

            response_list.append(code_entry)

            save_single_phase_log(engine, i, "Phase3", p3_prompt, f"Code: {code}\nThinking: {thinking}")
            save_single_phase_log(engine, i, "code", "", code)

            checkpoint_data = {
                'responses': response_list,
                'answer_Phase1': phase1_results,
                'answer_Phase2': phase2_results,
                'next_index': i + 1,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False)

        except Exception as e:
            print(f"Error at index {i}: {e}")
            db_id = db_path_list[i].split('/')[-1].split('.sqlite')[0]
            error_entry = {
                'question': question_list[i],
                'code': '',
                'thinking': f'Processing failed: {str(e)}',
                'db_id': db_id,
                'question_id': question_id_list[i],
                'error': str(e)
            }
            response_list.append(error_entry)

            checkpoint_data = {
                'responses': response_list,
                'answer_Phase1': phase1_results,
                'answer_Phase2': phase2_results,
                'next_index': i + 1,  
                'last_error': str(e)
            }
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False)

    print(f"Processing completed. Checkpoint saved to: {checkpoint_file}")

    return response_list, phase1_results, phase2_results

def decouple_question_schema(datasets, db_root_path):
    question_id_list = []
    question_list = []
    db_path_list = []
    knowledge_list = []
    gold_code_list = [] 

    for data in datasets:
        question_id_list.append(data['question_id'])
        question_list.append(data['question'])
        db_path_list.append(f"{db_root_path}{data['db_id']}/{data['db_id']}.sqlite")

        g_code = data.get('code')
        if isinstance(g_code, list): 
            g_code = g_code[0]
        gold_code_list.append(str(g_code))

        evidence = data.get('evidence', '')
        verified_knowledge = data.get('verified_knowledge', '')

        combined_knowledge = []
        if evidence:
            combined_knowledge.append(f"Evidence: {evidence}")
        if verified_knowledge:
            combined_knowledge.append(f"Verified Knowledge: {verified_knowledge}")

        knowledge_list.append(" | ".join(combined_knowledge) if combined_knowledge else "")

    return question_id_list, question_list, db_path_list, knowledge_list, gold_code_list

def generate_code_file(code_list, output_path=None):
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(code_list, f, indent=4, ensure_ascii=False)
    return code_list

if __name__ == '__main__':
    eval_path = '../Verified_Bird_Python.json'
    mode = 'dev'
    use_knowledge = 'True'
    db_root_path = '../dev_databases/'
    api_key = 'YOUR_API_KEY'
    engine = 'qwen3-max' 
    data_output_path = f'./exp_result/{engine}_output_kg_3phase' 
    limit_count = None
    print(f"Loading data from {eval_path}...")
    eval_data = json.load(open(eval_path, 'r', encoding="utf-8"))
    
    question_id_list, question_list, db_path_list, knowledge_list, gold_code_list = decouple_question_schema(
        datasets=eval_data,
        db_root_path=db_root_path
    )

    print("Starting 3-Phase Protocol...")
    print(f"Target Model: {engine}")
    print(f"Oracle Model: qwen-max")

    responses, p1_res, p2_res = collect_response_from_gpt_3phase(
        db_path_list=db_path_list,
        question_list=question_list,
        gold_code_list=gold_code_list, 
        api_key=api_key,
        engine=engine,
        knowledge_list=knowledge_list if use_knowledge == 'True' else None,
        limit_count=limit_count
    )
    
    output_name = os.path.join(data_output_path, f'predict_{mode}.json')
    generate_code_file(code_list=responses, output_path=output_name)
    
    full_log_path = os.path.join(data_output_path, f'full_log_{mode}.json')
    full_data = {
        str(i): {
            "question": question_list[i],
            "gold_code": gold_code_list[i],
            "answer_Phase1": p1_res[i] if i < len(p1_res) else "",
            "answer_Phase2": extract_oracle_answer(p2_res[i]) if i < len(p2_res) else "",
            "inquiry_category": extract_inquiry_category(p2_res[i]) if i < len(p2_res) else "",
            "final_code": responses[i]['code'] if i < len(responses) and isinstance(responses[i], dict) else ""
        }
        for i in range(len(responses))
    }
    with open(full_log_path, 'w', encoding='utf-8') as f:
        json.dump(full_data, f, indent=4, ensure_ascii=False)
    
    print(f'Successfully collected results from {engine} for {mode} evaluation')
    print(f'Results saved to: {output_name}')
    print(f'Full logs saved to: {full_log_path}')