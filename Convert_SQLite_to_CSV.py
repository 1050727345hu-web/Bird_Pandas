import os
import sqlite3
import pandas as pd
import csv
import sys

SOURCE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "dev_databases"))
TARGET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "excel_database"))

def get_primary_key(conn, table_name):
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info('{table_name}')")
    columns = cursor.fetchall()    
    pk_columns = [col[1] for col in columns if col[5] > 0]
    return pk_columns

def export_sqlite_to_csv(db_path, output_dir, db_id):
 
    db_output_dir = os.path.join(output_dir, db_id)
    os.makedirs(db_output_dir, exist_ok=True)
    
    print(f"Processing database: {db_id} -> {db_output_dir}")
    
    conn = sqlite3.connect(db_path)
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'")
        tables = [row[0] for row in cursor.fetchall()]
        
        for table in tables:
            print(f"  - Exporting table: {table}")
            
            pk_cols = get_primary_key(conn, table)
            if pk_cols:
                order_clause = f"ORDER BY {', '.join([f'`{c}`' for c in pk_cols])}"
            else:
                order_clause = "ORDER BY rowid"
            
            query = f"SELECT * FROM `{table}` {order_clause}"
            
            chunks = pd.read_sql_query(query, conn, chunksize=10000)
            
            csv_path = os.path.join(db_output_dir, f"{table}.csv")
            
            if os.path.exists(csv_path):
                os.remove(csv_path)
            
            has_data = False
            for i, chunk in enumerate(chunks):
                has_data = True
                mode = 'w' if i == 0 else 'a'
                header = (i == 0)
                
                chunk.to_csv(
                    csv_path,
                    mode=mode,
                    header=header,
                    index=False,
                    encoding='utf-8-sig', 
                    
                    float_format='%.10f', 
                    quoting=csv.QUOTE_NONNUMERIC 
                )
            
            if not has_data:
                cursor.execute(f"SELECT * FROM `{table}` LIMIT 0")
                col_names = [description[0] for description in cursor.description]
                pd.DataFrame(columns=col_names).to_csv(
                    csv_path, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC
                )
                
    except Exception as e:
        print(f"  !!! Error processing database {db_id}: {str(e)}")
    finally:
        conn.close()

def main():
     
    print(f"Source database directory: {SOURCE_DIR}")
    print(f"Target output directory: {TARGET_DIR}")
    
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory does not exist {SOURCE_DIR}")
        return

    count = 0
    items = sorted(os.listdir(SOURCE_DIR))
    
    for item in items:
        item_path = os.path.join(SOURCE_DIR, item)
        
        if os.path.isdir(item_path):
            db_id = item 
            
            sqlite_file = os.path.join(item_path, f"{item}.sqlite")
            
            if not os.path.exists(sqlite_file):
                candidates = [f for f in os.listdir(item_path) if f.endswith('.sqlite')]
                if len(candidates) == 1:
                    sqlite_file = os.path.join(item_path, candidates[0])
                elif len(candidates) > 1:
                    print(f"Warning: {db_id} has multiple SQLite files, skipping to avoid ambiguity: {candidates}")
                    continue
                else:
                    continue

            if os.path.exists(sqlite_file):
                export_sqlite_to_csv(sqlite_file, TARGET_DIR, db_id)
                count += 1

    print("\n" + "="*56)
    print(f"Processing completed. Processed {count} databases.")
    print(f"CSV files saved to: {TARGET_DIR}")

if __name__ == "__main__":
    main()
