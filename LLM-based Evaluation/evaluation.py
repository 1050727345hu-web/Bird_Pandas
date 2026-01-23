import sys
import json
import pandas as pd
import datetime
import io
import contextlib
import traceback
import numpy as np
import threading
import os
import gc
import psutil
import time
import re
import ast
import datetime as dt
from openai import OpenAI

QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_API_KEY = "YOUR_API_KEY"
def ensure_log_directory():
    log_dir = 'logs_dev'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

def log_error(error_message, error_type="ERROR"):
    ensure_log_directory()
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"\n[{timestamp}] {error_type}:\n{error_message}\n{'='*80}\n"
    
    with threading.Lock():
        with open('logs_dev/error_info.txt', 'a', encoding='utf-8') as f:
            f.write(log_message)
            f.flush()

def new_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

class LLMValidator:
    _instance = None
    _client = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if LLMValidator._client is None:
            print("Initializing LLMValidator (qwen-max-latest API)...")
            LLMValidator._client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)
            print("LLMValidator initialized successfully")

    @staticmethod
    def _force_type_match(value, target_type):
        try:
            if isinstance(target_type, type):
                if isinstance(value, target_type):
                    return value
                
                if target_type == int:
                    if isinstance(value, float):
                        if value.is_integer():
                            return int(value)
                    elif isinstance(value, str):
                        if value.isdigit():
                            return int(value)
                
                elif target_type == float:
                    if isinstance(value, (int, str)):
                        return float(value)
                
                elif target_type == str:
                    return str(value)
                
                elif target_type == list:
                    if not isinstance(value, (list, tuple)):
                        return [value]
        except:
            pass
        return value

    @staticmethod
    def _normalize_list_structure(lst):
        if not isinstance(lst, list):
            return lst
        
        # 检查是否所有元素类型相同
        types = set(type(x) for x in lst if x is not None)
        if len(types) == 1:
            target_type = types.pop()
            lst = [LLMValidator._force_type_match(x, target_type) if x is not None else None for x in lst]
        
        return lst

    @staticmethod
    def _recursive_convert(obj):
        if isinstance(obj, (list, tuple, np.ndarray)):
            return [LLMValidator._recursive_convert(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: LLMValidator._recursive_convert(v) for k, v in obj.items()}
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d')
        elif isinstance(obj, (np.generic, np.number)):
            return obj.item() if obj.size == 1 else obj.tolist()
        elif isinstance(obj, float) and np.isnan(obj):
            return None
        elif isinstance(obj, float):
            return round(obj, 2)
        elif isinstance(obj, str):
            try:
                if obj.isdigit():
                    return int(obj)
                elif obj.replace('.', '', 1).isdigit():
                    return round(float(obj), 2)
                elif '-' in obj or '/' in obj:
                    try:
                        date = pd.to_datetime(obj)
                        return date.strftime('%Y-%m-%d')
                    except:
                        pass
            except:
                pass
            return obj.strip()
        else:
            return obj

    @staticmethod
    def postprocess_result(raw_result):
        processed = raw_result
        if isinstance(processed, (pd.DataFrame, pd.Series)):
            if processed.empty:
                return []
            processed = processed.replace({np.nan: None})
            processed = processed.astype(object).where(pd.notnull(processed), None)
            processed = processed.values.tolist()
        
        elif isinstance(processed, np.ndarray):
            processed = processed.tolist()
        
        processed = LLMValidator._recursive_convert(processed)
        
        if isinstance(processed, list):
            processed = LLMValidator._normalize_list_structure(processed)
            if len(processed) == 1:
                if isinstance(processed[0], (int, float, str, type(None))):
                    processed = processed[0]
        
        if processed in ([], {}, None, np.nan):
            return []
        
        return processed

    def compare_values(self, output, expected):
        try:
            processed_output = self.postprocess_result(output)
            processed_expected = self.postprocess_result(expected)
            
            if type(processed_output) != type(processed_expected):
                processed_output = self._force_type_match(processed_output, type(processed_expected))
            
            if isinstance(processed_expected, list) and isinstance(processed_output, list):
                processed_output = self._normalize_list_structure(processed_output)
                processed_expected = self._normalize_list_structure(processed_expected)

            def _truncate_list(data, max_len=30):
                if isinstance(data, list):
                    return data[:max_len]
                return data

            truncated_output = _truncate_list(processed_output)
            truncated_expected = _truncate_list(processed_expected)
            
            messages = [
                {"role": "system", "content": """
You are a professional output-equivalence validator. Decide whether Output 1 and Output 2 are semantically and materially equivalent. Follow these rules strictly:

1) Ignore data formatting and presentation; focus only on the actual content/meaning. A stringified list is equivalent to the same real list structure.
2) Ignore pandas Index metadata (e.g., "Index([...], dtype='object')"); compare only computed results.
3) Numeric representation: 1.0 == 1, and strings like "1.0" equal 1.0 after numeric coercion.
4) List/row order does not matter. Compare as multisets, not sequences.
5) Nesting level does not matter if underlying elements are the same. E.g., ["a","b"] == [["a"],["b"]] == [[["a"]]] after flattening.
6) Parse stringified collections (lists/tuples/dicts) into real structures before comparing.
7) Ignore whitespace, newlines, and other formatting differences.
8) For tabular data, compare the data content, not the container format (CSV vs JSON).
9) Single-element list equivalence: ['apple'] == 'apple'; [123] == 123.
10) List vs non-list: When a list has exactly one element equal to the scalar, treat them as equivalent.

Additional clarifications to avoid false negatives (do not weaken correctness):
11) Label–value pairs vs values-only: If one output is (label, value) pairs (or dict-like rows) and the other is values-only, ignore labels and compare the values after type normalization and order-insensitive matching.
12) Superset/subset columns: If one table has extra descriptive columns (e.g., names, titles, labels, text) and the other has only the target columns, consider them equivalent if a projection of the wider result exactly matches the narrower result (row-wise, order-insensitive). Extra columns may be ignored; missing required values may not.
13) Duplicates: If one output contains duplicate rows/values while the other contains each value once, compare after de-duplicating both sides. Duplicates alone must not cause inequality.
14) Orientation/shape: Nx1, 1xN, and nested forms that contain the same set of scalars are equivalent after flattening and (if needed) transposing.
15) Boolean normalization: Normalize and treat as equivalent true/false, "TRUE"/"FALSE", "yes"/"no", "YES"/"NO", "y"/"n", and 1/0.
16) Name tokenization: "First Last" is equivalent to ["First","Last"] when tokenizing by whitespace yields the same tokens (case-insensitive).
17) Numeric scale: Fractions and percentages are equivalent if multiplying/dividing by 100 aligns them within normal rounding tolerance (e.g., 0.2272727 == 22.7272727%). Normalize strings like "22.7%".
18) Date/time normalization: Normalize formats (e.g., "7:00" == "07:00:00"). Treat NaN/None/NULL/"NaN" as the same null. Apply rule 13 for repeated timestamps.
19) Natural-language sentences vs structured tuples: If a sentence unambiguously contains the same entities and numbers as the structured output, treat them as equivalent after extracting those entities and values.
20) No partial matches: After applying all normalizations, only mark as equivalent if both outputs express the same set of values. Do not infer missing values.

Respond with exactly one word: "Correct" if equivalent, "Incorrect" otherwise.
                """},
                {"role": "user", "content": f"Output 1: {truncated_output}\nOutput 2: {truncated_expected}"}
            ]
            
            try:
                response_obj = LLMValidator._client.chat.completions.create(
                    model="qwen3-coder-flash",
                    messages=messages,
                    max_tokens=7000,
                    temperature=0,
                    extra_body={"enable_thinking": True}
                )
                response = response_obj.choices[0].message.content.strip()
            except Exception as e:
                log_error(f"Model invocation error: {str(e)}")
                return 0, "error: " + str(e)

            is_correct = "Correct" in response and "Incorrect" not in response

            if not is_correct:
                log_error(f"""
Compare the error messages:
Test sequence number: {getattr(self, 'current_idx', 'Unknown')}
Large model response: {response}
output type: {type(output).__name__}
expected type: {type(expected).__name__}
Raw output: {output}
Original expectation: {expected}
Output: {processed_output}
Expected: {processed_expected}
Relevant code: {getattr(self, 'current_code', 'Unknown')}
""", "COMPARISON_INFO")

            return (1 if is_correct else 0), response
                
        except Exception as e:
            log_error(f"Comparison error: {str(e)}")
            return 0, "error: " + str(e)

def process_result(result):
    try:
        if isinstance(result, (pd.DataFrame, pd.Series, pd.Index, np.ndarray, list, tuple)):
            result = LLMValidator.postprocess_result(result)
        
        if isinstance(result, str):
            try:
                import ast
                result = ast.literal_eval(result)
                result = LLMValidator.postprocess_result(result)
            except:
                result = result.strip()
        
        return result
    except Exception as e:
        log_error(f"Result handling errors: {str(e)}")
        return result

def load_tables_for_db(db_id):
    tables = {}
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        excel_root = os.path.join(os.path.dirname(base_dir), 'excel_database')
        data_dir = os.path.join(excel_root, db_id) if db_id else ''
        if not data_dir or not os.path.isdir(data_dir):
            return tables
        for name in os.listdir(data_dir):
            if not name.lower().endswith('.csv'):
                continue
            path = os.path.join(data_dir, name)
            table = os.path.splitext(name)[0]
            try:
                try:
                    df = pd.read_csv(path, low_memory=False, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(path, low_memory=False, encoding='gbk')
                tables[table] = df
            except Exception:
                continue
    except Exception:
        pass
    return tables

def replace_read_csv_with_tables(code_str, tables):
    modified = code_str
    for table in tables.keys():
        pattern = re.compile(
            rf"(?:pd|pandas)\.read_csv\s*\(\s*"
            rf"['\"][^'\"]*{re.escape(table)}\.csv['\"]\s*"
            rf"(?:,[^)]*)?\)",
            flags=re.IGNORECASE
        )
        modified = pattern.sub(f"{table}_df", modified)
    return modified

def tuple_to_list(o):
    if o is None or isinstance(o, (bool, int, float, str)):
        return o
    if isinstance(o, (pd.Timestamp, dt.datetime, dt.date)):
        try:
            return o.isoformat()
        except Exception:
            return str(o)
    if isinstance(o, np.datetime64):
        try:
            return pd.Timestamp(o).isoformat()
        except Exception:
            return str(o)
    if o is pd.NaT:
        return None
    if isinstance(o, np.generic):
        try:
            return tuple_to_list(o.item())
        except Exception:
            return str(o)
    if isinstance(o, pd.DataFrame):
        return [tuple_to_list(row) for row in o.values.tolist()]
    if isinstance(o, pd.Series):
        return [tuple_to_list(x) for x in o.tolist()]
    if isinstance(o, np.ndarray):
        try:
            return [tuple_to_list(x) for x in o.tolist()]
        except Exception:
            return str(o)
    if isinstance(o, (tuple, set)):
        return [tuple_to_list(x) for x in o]
    if isinstance(o, list):
        return [tuple_to_list(x) for x in o]
    if isinstance(o, dict):
        return {str(k): tuple_to_list(v) for k, v in o.items()}
    try:
        json.dumps(o)
        return o
    except TypeError:
        return str(o)

def to_rows(value):
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        return tuple_to_list(value)
    if isinstance(value, pd.Series):
        data = value.tolist()
        return [[tuple_to_list(x)] for x in data]
    if isinstance(value, np.ndarray):
        try:
            data = value.tolist()
        except Exception:
            return [[tuple_to_list(str(value))]]
        if isinstance(data, list) and data and isinstance(data[0], list):
            return [tuple_to_list(row) for row in data]
        return [[tuple_to_list(x)] for x in data]
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda kv: str(kv[0]))
        return [[tuple_to_list(k), tuple_to_list(v)] for k, v in items]
    if isinstance(value, (list, tuple)):
        seq = list(value)
        if not seq:
            return []
        if isinstance(seq[0], (list, tuple)):
            return [tuple_to_list(list(row)) for row in seq]
        return [[tuple_to_list(x)] for x in seq]
    return [[tuple_to_list(value)]]

def execute_code(code, db_id, expected_output, validator, code_idx=None):
    validator.current_idx = code_idx
    validator.current_code = code

    tables = load_tables_for_db(db_id) if db_id else {}

    exec_globals = {
        'pd': pd,
        'pandas': pd,
        'np': np,
        'numpy': np,
        '__builtins__': __builtins__,
    }
    for tname, df in tables.items():
        exec_globals[tname] = df
        exec_globals[f'{tname}_df'] = df

    code_run = replace_read_csv_with_tables(code, tables) if tables else code

    pre_keys = set(exec_globals.keys())
    stdout_main = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_main):
            exec(code_run, exec_globals)
    except Exception:
        code_value = None
    else:
        code_value = None
        try:
            function_names = []
            func_pattern = r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
            try:
                function_names.extend(re.findall(func_pattern, code_run))
            except Exception:
                pass
            try:
                tree_for_funcs = ast.parse(code_run)
                for node in ast.walk(tree_for_funcs):
                    if isinstance(node, ast.FunctionDef) and not node.name.startswith('__'):
                        function_names.append(node.name)
            except Exception:
                pass
            function_names = list(dict.fromkeys(function_names))

            got_result = False
            for fname in function_names:
                if got_result:
                    break
                obj = exec_globals.get(fname)
                if obj is None or not callable(obj):
                    continue
                try:
                    import inspect
                    sig = inspect.signature(obj)
                    params = list(sig.parameters.keys())
                    args = []
                    used_tables = set()
                    for param in params:
                        param_lower = param.lower()
                        param_base = param_lower.replace('_df', '')
                        matched = False
                        for tname in tables.keys():
                            t_lower = tname.lower()
                            if (
                                t_lower == param_lower or
                                t_lower == param_base or
                                t_lower in param_lower or
                                param_lower in t_lower
                            ) and tname not in used_tables:
                                args.append(tables[tname])
                                used_tables.add(tname)
                                matched = True
                                break
                        if not matched:
                            for tname in tables.keys():
                                if tname not in used_tables:
                                    args.append(tables[tname])
                                    used_tables.add(tname)
                                    matched = True
                                    break
                    func_stdout = io.StringIO()
                    if len(params) == 0:
                        with contextlib.redirect_stdout(func_stdout):
                            result = obj()
                        if result is not None:
                            code_value = result
                            got_result = True
                        elif func_stdout.getvalue().strip():
                            code_value = func_stdout.getvalue()
                            got_result = True
                    elif len(args) == len(params):
                        with contextlib.redirect_stdout(func_stdout):
                            result = obj(*args)
                        if result is not None:
                            code_value = result
                            got_result = True
                        elif func_stdout.getvalue().strip():
                            code_value = func_stdout.getvalue()
                            got_result = True
                    else:
                        available_tables = list(tables.values())
                        if len(params) <= len(available_tables) and len(available_tables) > 0:
                            with contextlib.redirect_stdout(func_stdout):
                                result = obj(*available_tables[:len(params)])
                            if result is not None:
                                code_value = result
                                got_result = True
                            elif func_stdout.getvalue().strip():
                                code_value = func_stdout.getvalue()
                                got_result = True
                except Exception:
                    continue
        except Exception:
            pass

        if code_value is None:
            if 'result' in exec_globals:
                code_value = exec_globals['result']
            else:
                for key in ('output', 'final_result', 'res', 'answer', 'data', 'df'):
                    if key in exec_globals:
                        code_value = exec_globals[key]
                        break

        if code_value is None:
            try:
                tree = ast.parse(code_run)
                last_var = None
                last_expr = None
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign) and node.targets:
                        target = node.targets[0]
                        if isinstance(target, ast.Name):
                            last_var = target.id
                    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                        last_var = node.target.id
                if hasattr(tree, 'body') and tree.body:
                    for n in reversed(tree.body):
                        if isinstance(n, ast.Expr):
                            last_expr = n
                            break
                if last_var and last_var in exec_globals:
                    code_value = exec_globals[last_var]
                elif last_expr is not None:
                    try:
                        compiled = compile(ast.Expression(last_expr.value), '<ast>', 'eval')
                        eval_stdout = io.StringIO()
                        with contextlib.redirect_stdout(eval_stdout):
                            value = eval(compiled, exec_globals)
                        if value is not None:
                            code_value = value
                        elif eval_stdout.getvalue().strip():
                            code_value = eval_stdout.getvalue()
                    except Exception:
                        pass
            except Exception:
                pass

        if code_value is None:
            typical_names = (
                'filtered_result', 'merged_sorted', 'filtered', 'merged',
                'final_df', 'final', 'answer_df', 'top_record'
            )
            for name in typical_names:
                if name in exec_globals:
                    val = exec_globals[name]
                    if isinstance(val, (pd.DataFrame, pd.Series, np.ndarray, list, tuple, dict)):
                        code_value = val
                        break
        if code_value is None:
            exclude_keys = set(pre_keys)
            exclude_keys.update(tables.keys())
            exclude_keys.update({f'{t}_df' for t in tables.keys()})
            candidates = [
                (k, exec_globals[k]) for k in exec_globals.keys()
                if k not in exclude_keys and not k.startswith('__')
            ]
            for k, v in candidates:
                if isinstance(v, (pd.DataFrame, pd.Series)):
                    code_value = v
                    break
            if code_value is None:
                for k, v in candidates:
                    if isinstance(v, (np.ndarray, list, tuple, dict, np.generic, str, int, float, bool)):
                        code_value = v
                        break

        if code_value is None:
            captured = stdout_main.getvalue().strip()
            if captured:
                code_value = captured

    actual_result = to_rows(code_value)
    if code and actual_result is None:
        actual_result = []

    comparison_result, llm_response = validator.compare_values(actual_result, expected_output)
    return actual_result, comparison_result, llm_response

def write_comparison_result(result_dict, output_file='./logs_dev/comparison_results.jsonl'):
    ensure_log_directory()
    with threading.Lock():
        try:
            existing_code_idxs = set()
            if os.path.exists(output_file):
                with open(output_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                existing_result = json.loads(line)
                                existing_code_idxs.add(existing_result.get('code_idx'))
                            except json.JSONDecodeError:
                                continue
            
            current_code_idx = result_dict.get('code_idx')
            if current_code_idx in existing_code_idxs:
                log_error(f"Skip duplicate writes; code_idx {current_code_idx} already exists", "DUPLICATE_WRITE_WARNING")
                return
            
            serializable_dict = {}
            for key, value in result_dict.items():
                if isinstance(value, (np.ndarray, pd.DataFrame, pd.Series)):
                    serializable_dict[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    serializable_dict[key] = value.item()
                else:
                    serializable_dict[key] = value
            
            json_str = json.dumps(serializable_dict, ensure_ascii=False)
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json_str + '\n')
                f.flush()
        except Exception as e:
            log_error(f"Error writing comparison result: {str(e)}\n result data: {result_dict}")

def read_comparison_results(output_file='./logs_dev/comparison_results.jsonl'):

    results = []
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        log_error(f"JSON parse error: {str(e)}\n Problem line: {line}")
                        continue
    return results

def execute_model(code_data, question_id, meta_time_out, expected_outputs):
    try:
        code = code_data['code']
        db_id = code_data['db_id']
        expected_output = next(
            (item.get('output') for item in expected_outputs  
            if item.get('question_id') == question_id),
            None
        )
        
        if expected_output is None:
            log_error(f"""
Execution error:
Test number: {question_id}
Error: The expected output cannot be found
The code:
{code}
""", "EXECUTION_ERROR")
            result = {
                'code_idx': question_id,
                'res': 0,
                'actual_output': None,
                'expected_output': None,
                'error': 'The expected output could not be found'
            }
            write_comparison_result(result)
            return result
        
        validator = LLMValidator.get_instance()
        actual_output, comparison_result, llm_response = execute_code(code, db_id, expected_output, validator, question_id)
        print(f"Comparison result for idx {question_id}: {comparison_result}")
        
        result = {
            'code_idx': question_id,
            'res': comparison_result,
            'actual_output': actual_output,
            'expected_output': expected_output,
            'db_id': db_id,  
            'llm_response': llm_response  
        }
        write_comparison_result(result)
        
        gc.collect()
        
        return result
        
    except Exception as e:  
        error_msg = str(e)
        log_error(f"""
Execution error:
Test number: {question_id}
Error: {error_msg}
Code:
{code}
""", "EXECUTION_ERROR")
        result = {
            'code_idx': question_id,
            'res': 0,
            'actual_output': None,
            'expected_output': expected_output if 'expected_output' in locals() else None,
            'error': error_msg
        }
        write_comparison_result(result)
        return result

def package_codes(code_path, data_mode='dev'):
    code_file = "./exp_result/predict_dev.json"
    try:
        with open(code_file, 'r', encoding='utf-8') as f:
            code_data = json.load(f)
            if isinstance(code_data, dict):
                code_data = [code_data]
            
            for i, item in enumerate(code_data):
                if 'question_id' not in item and 'idx' not in item:
                    item['question_id'] = i
            
            seen_question_ids = set()
            unique_code_data = []
            duplicate_count = 0
            
            for item in code_data:
                question_id = item.get('question_id', item.get('idx'))
                if question_id not in seen_question_ids:
                    seen_question_ids.add(question_id)
                    unique_code_data.append(item)
                else:
                    duplicate_count += 1
                    log_error(f"Duplicate question_id: {question_id}，skipped", "DUPLICATE_WARNING")
            
            if duplicate_count > 0:
                print(f"Warning: found and removed {duplicate_count} duplicate records")
                print(f"Original data: {len(code_data)} records, after deduplication: {len(unique_code_data)} records")
                log_error(f"Total {duplicate_count} duplicate records, original data {len(code_data)} records, after deduplication {len(unique_code_data)} records", "DEDUPLICATION_INFO")
            
            return unique_code_data
            
    except FileNotFoundError:
        raise
    except json.JSONDecodeError:
        raise

def clean_memory():
    gc.collect()
    
    process = psutil.Process()
    memory_info = process.memory_info()
    
    memory_used = memory_info.rss / 1024 / 1024
    return memory_used

def run_sequential(code_data, expected_outputs, meta_time_out=30.0, batch_size=10):
    if not code_data:
        print("Error: No code data to process")
        return []
        
    print(f"Processing {len(code_data)} test cases...")
    
    output_file = './logs_dev/comparison_results.jsonl'
    if os.path.exists(output_file):
        os.remove(output_file)
    
    validator = LLMValidator.get_instance()
    
    for batch_start in range(0, len(code_data), batch_size):
        batch_end = min(batch_start + batch_size, len(code_data))
        batch = code_data[batch_start:batch_end]
        
        for i, code_item in enumerate(batch):
            question_id = code_item.get('question_id', code_item.get('idx'))
            if question_id is None:
                log_error(f"Missing question_id in code_data: {code_item}")
                continue
                
            execute_model(code_item, question_id, meta_time_out, expected_outputs)
            #print(f"\rProgress: [{batch_start + i + 1}/{len(code_data)}] cases processed", end='', flush=True)
        
        gc.collect()
        time.sleep(1)  
    
    print(f"\nExecution completed. Reading results...")
    results = read_comparison_results()
    return sort_results(results)

def sort_results(list_of_dicts):
    return sorted(list_of_dicts, key=lambda x: x['code_idx'])

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compute_acc_by_diff(exec_results, diff_json_path):
    contents = load_json(diff_json_path)
    
    simple_results = []
    moderate_results = []
    challenging_results = []
    simple_correct = 0
    moderate_correct = 0
    challenging_correct = 0
    
    difficulty_map = {item['question_id']: item['difficulty'] for item in contents}
    
    for result in exec_results:
        idx = result['code_idx']
        if idx in difficulty_map:
            difficulty = difficulty_map[idx]
            is_correct = result['res'] == 1
            
            if difficulty == 'simple':
                simple_results.append(result)
                if is_correct:
                    simple_correct += 1
            elif difficulty == 'moderate':
                moderate_results.append(result)
                if is_correct:
                    moderate_correct += 1
            elif difficulty == 'challenging':
                challenging_results.append(result)
                if is_correct:
                    challenging_correct += 1
    
    def calculate_accuracy(correct, total):
        return (correct / total * 100) if total > 0 else 0.0
    
    simple_acc = calculate_accuracy(simple_correct, len(simple_results))
    moderate_acc = calculate_accuracy(moderate_correct, len(moderate_results))
    challenging_acc = calculate_accuracy(challenging_correct, len(challenging_results))
    
    total_correct = simple_correct + moderate_correct + challenging_correct
    total_count = len(simple_results) + len(moderate_results) + len(challenging_results)
    
    total_acc = (total_correct / total_count) * 100 if total_count > 0 else 0.0
      
    count_lists = [
        len(simple_results),
        len(moderate_results),
        len(challenging_results),
        total_count
    ]
    
    return simple_acc, moderate_acc, challenging_acc, total_acc, count_lists

def print_data(score_lists, count_lists):
    levels = ['simple', 'moderate', 'challenging', 'total']
    
    result_summary = "\n" + "="*100 + "\n"
    result_summary += "EVALUATION RESULTS\n"
    result_summary += "="*100 + "\n\n"
    
    result_summary += "TEST CASES COUNT:\n"
    result_summary += "-"*100 + "\n"
    result_summary += "{:20} {:20} {:20} {:20} {:20}\n".format("Difficulty", *levels)
    result_summary += "{:20} {:<20} {:<20} {:<20} {:<20}\n".format("Count", *count_lists)
    
    result_summary += "\nACCURACY RESULTS:\n"
    result_summary += "-"*100 + "\n"
    result_summary += "{:20} {:20} {:20} {:20} {:20}\n".format("Difficulty", *levels)
    result_summary += "{:20} {:<20.4f}% {:<20.4f}% {:<20.4f}% {:<20.4f}%\n".format(
        "Accuracy", *score_lists)
    
    result_summary += "\nDETAILED STATISTICS:\n"
    result_summary += "-"*100 + "\n"
    total_correct = 0
    for i, level in enumerate(levels[:-1]):
        correct = int((score_lists[i] * count_lists[i]) / 100)
        if level != 'total':
            total_correct += correct
        result_summary += f"{level:12}: {correct:3d}/{count_lists[i]:<3d} correct, " \
                          f"accuracy = {score_lists[i]:.4f}%\n"
    
    result_summary += "-"*50 + "\n"
    result_summary += f"{'total':12}: {total_correct:3d}/{count_lists[-1]:<3d} correct, " \
                      f"accuracy = {(total_correct * 100 / count_lists[-1]):.4f}%\n"
    
    result_summary += "\n" + "="*100 + "\n"
    
    print(result_summary)
    
    with open('logs_max_dev/error_info.txt', 'a', encoding='utf-8') as f:
        f.write(result_summary)

if __name__ == '__main__':
    PREDICTED_CODE_PATH = './data'
    DATA_MODE = 'dev'
    META_TIME_OUT = 30.0
    DIFF_JSON_PATH = '../Verified_Bird_Python.json'
    BATCH_SIZE = 1
    OUTPUT_FILE = './logs_dev/comparison_results.jsonl'

    code_data = package_codes(PREDICTED_CODE_PATH, DATA_MODE)
    expected_outputs = load_json(DIFF_JSON_PATH)
    
    exec_results = run_sequential(
        code_data, 
        expected_outputs,
        meta_time_out=META_TIME_OUT,
        batch_size=BATCH_SIZE
    )
    
    simple_acc, moderate_acc, challenging_acc, acc, count_lists = \
        compute_acc_by_diff(exec_results, DIFF_JSON_PATH)
    
    score_lists = [simple_acc, moderate_acc, challenging_acc, acc]
    print_data(score_lists, count_lists)
