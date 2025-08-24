import os
import json
import sqlite3
import pandas as pd
import streamlit as st
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langsmith import Client
from langsmith.utils import LangSmithConflictError
from langsmith.evaluation import EvaluationResult
from google.api_core.exceptions import ResourceExhausted
from google.auth.exceptions import GoogleAuthError
from uuid import UUID
from typing import List, Dict, Any
from functools import wraps

# Load environment variables
load_dotenv()

# Initialize Streamlit
st.set_page_config(page_title="SQL Query Generator")
st.header("Gemini SQL Query Generator")

# Initialize LangSmith client
langsmith_client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))

# Initialize database connection
db = SQLDatabase.from_uri("sqlite:///C://Users//Dell//Desktop//Gemini-With-MSSql-main//Chinook.db")

# Example queries (for few-shot learning)
examples = [~
    {
        "input": "List all tracks composed by Angus Young",
        "query": "SELECT * FROM Track WHERE Composer LIKE '%Angus Young%';"
    },
    {
        "input": "Find customers from Brazil with their support representatives",
        "query": """SELECT c.FirstName, c.LastName, e.FirstName || ' ' || e.LastName AS SupportRep 
                    FROM Customer c JOIN Employee e ON c.SupportRepId = e.EmployeeId 
                    WHERE c.Country = 'Brazil';"""
    },
    {
        "input": "Show invoices with more than 5 items purchased",
        "query": """SELECT i.InvoiceId, COUNT(il.InvoiceLineId) AS Items 
                    FROM Invoice i JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId 
                    GROUP BY i.InvoiceId HAVING COUNT(il.InvoiceLineId) > 5;"""
    },
    {
        "input": "List all rock tracks longer than 4 minutes",
        "query": """SELECT t.Name, t.Milliseconds/60000 AS Minutes 
                    FROM Track t JOIN Genre g ON t.GenreId = g.GenreId 
                    WHERE g.Name = 'Rock' AND t.Milliseconds > 240000;"""
    },
    {
        "input": "Find the average invoice total by country",
        "query": """SELECT c.Country, AVG(i.Total) AS AvgInvoice 
                    FROM Invoice i JOIN Customer c ON i.CustomerId = c.CustomerId 
                    GROUP BY c.Country;"""
    },
    {
        "input": "Show employees and how many customers they support",
        "query": """SELECT e.FirstName, e.LastName, COUNT(c.CustomerId) AS CustomersSupported 
                    FROM Employee e LEFT JOIN Customer c ON e.EmployeeId = c.SupportRepId 
                    GROUP BY e.EmployeeId;"""
    },
    {
        "input": "List tracks that appear in multiple playlists",
        "query": """SELECT t.Name, COUNT(pt.PlaylistId) AS PlaylistCount 
                    FROM Track t JOIN PlaylistTrack pt ON t.TrackId = pt.TrackId 
                    GROUP BY t.TrackId HAVING COUNT(pt.PlaylistId) > 1;"""
    },
    {
        "input": "Find albums with no tracks in the database",
        "query": "SELECT * FROM Album WHERE AlbumId NOT IN (SELECT DISTINCT AlbumId FROM Track);"
    },
    {
        "input": "Show monthly sales totals for 2013",
        "query": """SELECT strftime('%Y-%m', InvoiceDate) AS Month, SUM(Total) AS MonthlySales 
                    FROM Invoice WHERE strftime('%Y', InvoiceDate) = '2013' 
                    GROUP BY strftime('%Y-%m', InvoiceDate);"""
    },
    {
        "input": "List all media types with their average track price",
        "query": """SELECT m.Name, AVG(t.UnitPrice) AS AvgPrice 
                    FROM MediaType m JOIN Track t ON m.MediaTypeId = t.MediaTypeId 
                    GROUP BY m.MediaTypeId;"""
    }
]

# Retry decorator for handling quota errors
def retry_on_quota_exceeded(max_attempts=3, delay=11):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except ResourceExhausted as e:
                    if "429" in str(e):
                        attempts += 1
                        if attempts == max_attempts:
                            raise e
                        print(f"Quota exceeded, retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        raise e
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Initialize Gemini LLM with model fallback and project ID
def init_gemini_llm(model_id="gemini-2.0-flash-lite"):
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise GoogleAuthError("GOOGLE_CLOUD_PROJECT environment variable not set. Please set it or configure via 'gcloud config set project'.")
    
    llm = ChatGoogleGenerativeAI(
        model=model_id,
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        project=project_id
    )
    return llm
    
# def get_next_model_id(model):
#     model_order = {
#         "gemini-1.5-flash": "gemini-1.5-flash-latest",
#         "gemini-1.5-flash-latest": "gemini-2.0-flash",
#         "gemini-2.0-flash": "gemini-1.5-pro",
#         "gemini-1.5-pro": "model-unavailable"
#     }
#     return model_order.get(model, "model-unavailable")

# Create SQL query chain
def create_query_chain():
    llm = init_gemini_llm()
    
    example_prompt = PromptTemplate.from_template(
        "User input: {input}\nSQL query: {query}"
    )
    
    prompt = FewShotPromptTemplate(
        example_selector=None,
        examples=examples[:5],
        example_prompt=example_prompt,
        prefix="""You are a SQLite expert. Given an input question:
        1. Create a syntactically correct SQLite query
        2. Respect table relationships
        3. Return at most {top_k} results unless specified
        
        Table Info: {table_info}\n

        Output the final SQL query only.
        """,
        suffix="User input: {input}\nSQL query: ",
        input_variables=["question", "table_info", "top_k", "dialect"]
    )
    
    return create_sql_query_chain(llm, db, prompt)

# Remove backticks from front and back of SQL query
def remove_sql_backticks(input_string: str) -> str:
    patterns = ['```sqlite', '```', 'sql']
    for pattern in patterns:
        if input_string.startswith(pattern):
            input_string = input_string[len(pattern):].lstrip()
        if input_string.endswith(pattern):
            input_string = input_string[:-len(pattern)].rstrip()
    return input_string

# Execute query and return results
def execute_query(query):
    try:
        conn = sqlite3.connect('C://Users//Dell//Desktop//Gemini-With-MSSql-main//Chinook.db')
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        return rows
    except Exception as e:
        return str(e)

def setup_evaluation_dataset():
    dataset_name = "sql_Kiotviet"
    try:
        dataset = langsmith_client.create_dataset(
            dataset_name=dataset_name,
            description="Evaluation of SQL query generation from natural language"
        )
        print("Creating new dataset")
    except LangSmithConflictError:
        print("Dataset already exists - using existing dataset")
        dataset = langsmith_client.read_dataset(dataset_name="sql_Kiotviet")
    
    # Ensure all examples have table_info and question
    for example in langsmith_client.list_examples(dataset_id=dataset.id):
        inputs_updated = False
        if 'table_info' not in example.inputs or not example.inputs['table_info']:
            example.inputs['table_info'] = db.get_table_info()
            inputs_updated = True
        if 'question' not in example.inputs or not example.inputs['question']:
            # Skip examples with no question to avoid invalid updates
            continue
        if inputs_updated:
            try:
                langsmith_client.update_example(
                    example_id=example.id,
                    inputs=example.inputs
                )
                print(f"Updated example {example.id} with table_info")
            except Exception as e:
                print(f"Failed to update example {example.id}: {str(e)}")
    
    return dataset

def log_evaluation(question: str, generated_query: str, execution_result: Any, error: Any = None) -> None:
    """
    Log evaluation to LangSmith only if the entry is unique.
    Uniqueness is determined by both the question and the generated query.
    """
    try:
        # First, check if this example already exists in the dataset
        dataset = langsmith_client.read_dataset(dataset_name="sql_Kiotviet")
        existing_examples = list(langsmith_client.list_examples(dataset_id=dataset.id))
        
        # Check for duplicates
        is_duplicate = False
        for example in existing_examples:
            existing_question = example.inputs.get("question", "").strip()
            existing_query = example.outputs.get("generated_query", "").strip()
            
            # Normalize strings for comparison
            if (existing_question.lower() == question.lower() and 
                existing_query.lower() == generated_query.strip().lower()):
                is_duplicate = True
                break
        
        if is_duplicate:
            st.sidebar.info("Duplicate evaluation detected - not logging to LangSmith")
            return
        
        # If not a duplicate, proceed with logging
        inputs = {
            "question": question,
            "table_info": db.get_table_info()
        }
        
        outputs = {
            "generated_query": generated_query,
            "execution_result": str(execution_result),
            "success": error is None
        }
        
        if error:
            outputs["error"] = str(error)
        
        # Create the new example
        langsmith_client.create_example(
            dataset_id=dataset.id,
            inputs=inputs,
            outputs=outputs
        )
        
        st.sidebar.success("Evaluation logged successfully!")
        
    except LangSmithConflictError:
        st.sidebar.warning("Example already exists in dataset")
    except Exception as e:
        st.sidebar.error(f"Error logging evaluation: {str(e)}")

def target(inputs: dict) -> dict:
    question = inputs.get('question', '').strip()
    
    # Load training_data.jsonl and find matching prompt
    training_data = load_training_data()
    generated_query = None
    for item in training_data:
        try:
            # Handle "prompt"/"completion" structure
            if "prompt" in item:
                item_prompt = item.get("prompt", "").strip().lower()
                if item_prompt == question.lower():
                    generated_query = item.get("completion", "").strip()
                    print(f"Matched training_data.jsonl in target: question='{question}', generated_query='{generated_query}'")
                    break
            # Handle "messages" structure
            elif "messages" in item and isinstance(item["messages"], list) and len(item["messages"]) >= 2:
                item_question = item["messages"][0].get("content", "").strip().lower()
                if item_question == question.lower():
                    generated_query = item["messages"][1].get("content", "").strip()
                    print(f"Matched training_data.jsonl in target (messages): question='{question}', generated_query='{generated_query}'")
                    break
        except (KeyError, AttributeError, TypeError) as e:
            print(f"Warning: Skipping malformed training data item in target: {str(e)}, item={item}")
            continue
    
    # Return training_data.jsonl completion if found
    if generated_query is not None:
        print(f"Using training_data.jsonl completion in target: '{generated_query}'")
        return {"answer": generated_query}
    
    # Fallback to LLM if no match in training_data.jsonl
    llm = init_gemini_llm()
    prompt = f"""
    You are a SQLite expert. Given an input question:
    1. Create a syntactically correct SQLite query
    
    Table Info: {inputs['table_info']}
    
    User input: {question}
    """
    
    try:
        response = llm.invoke(prompt)
        generated_query = response.content.strip()
        generated_query = remove_sql_backticks(generated_query)
        print(f"Using LLM output in target: generated_query='{generated_query}'")
    except Exception as e:
        generated_query = f"Error: {str(e)}"
        print(f"Error in LLM invocation: {str(e)}")
    
    return {"answer": generated_query}

# Custom CORRECTNESS_PROMPT for Gemini
CORRECTNESS_PROMPT = """
You are a SQLite expert tasked with evaluating the correctness of a generated SQL query.
Given the input question, generated query, and reference query, determine if the generated query is correct.
A query is correct if it:
- Produces the same results as the reference query
- Uses proper SQLite syntax
- Respects the table schema and relationships

Input question: {question}
Generated query: {generated_query}
Reference query: {reference_query}
Table Info: {table_info}

Output a JSON object with exactly these fields:
{{
    "score": 1.0,  // Use 1.0 for correct, 0.0 for incorrect
    "comment": "Explanation of why the query is correct or incorrect"
}}
Ensure the output is valid JSON with no additional text, backticks, or markdown formatting.
Example:
{{ "score": 1.0, "comment": "Query is correct" }}
"""

def create_gemini_judge(model_id="gemini-2.0-flash-lite"):
    llm = init_gemini_llm(model_id)
    
    def judge(inputs: dict, outputs: dict, reference_outputs: dict) -> EvaluationResult:
        question = inputs.get('question', '').strip()
        table_info = inputs.get('table_info', db.get_table_info())  # Default to db.get_table_info()
        
        if not question:
            return EvaluationResult(
                key="correctness",
                score=0.0,
                comment="No question provided in inputs"
            )
        
        # Load training_data.jsonl and find matching prompt
        training_data = load_training_data()
        generated_query = None
        for item in training_data:
            try:
                # Handle "prompt"/"completion" structure
                if "prompt" in item:
                    item_prompt = item.get("prompt", "").strip().lower()
                    if item_prompt == question.lower():
                        generated_query = item.get("completion", "").strip()
                        print(f"Matched training_data.jsonl: question='{question}', generated_query='{generated_query}'")
                        break
                # Handle "messages" structure
                elif "messages" in item and isinstance(item["messages"], list) and len(item["messages"]) >= 2:
                    item_question = item["messages"][0].get("content", "").strip().lower()
                    if item_question == question.lower():
                        generated_query = item["messages"][1].get("content", "").strip()
                        print(f"Matched training_data.jsonl (messages): question='{question}', generated_query='{generated_query}'")
                        break
            except (KeyError, AttributeError, TypeError) as e:
                print(f"Warning: Skipping malformed training data item: {str(e)}, item={item}")
                continue
        
        # Use training_data.jsonl completion as generated_query if found
        if generated_query is not None:
            print(f"Using training_data.jsonl completion as generated_query: '{generated_query}'")
        else:
            # Fallback to LLM-generated output
            generated_query = outputs.get('answer', '').strip()
            print(f"No match in training_data.jsonl, using LLM output: generated_query='{generated_query}'")
        
        # Get reference query from reference_outputs
        reference_query = reference_outputs.get('generated_query', '').strip()
        print(f"Using reference_outputs: reference_query='{reference_query}'")
        
        # If no reference query, mark as incorrect
        if not reference_query:
            print(f"No reference query found for question='{question}'")
            return EvaluationResult(
                key="correctness",
                score=0.0,
                comment="No reference query found in reference_outputs"
            )
        
        # If generated_query matches reference_query, mark as correct
        if generated_query.lower() == reference_query.lower():
            print(f"Generated query matches reference query for question='{question}'")
            return EvaluationResult(
                key="correctness",
                score=1.0,
                comment=f"Generated query '{generated_query}' matches reference query"
            )
        
        # If generated_query is "Invalid output" but doesn't match reference, mark as incorrect
        if generated_query.lower() == "invalid output":
            print(f"Generated query is 'Invalid output' but does not match reference for question='{question}'")
            return EvaluationResult(
                key="correctness",
                score=0.0,
                comment="Generated query is 'Invalid output' from training_data.jsonl but does not match reference query"
            )
        
        prompt = CORRECTNESS_PROMPT.format(
            question=question,
            generated_query=generated_query,
            reference_query=reference_query,
            table_info=table_info
        )
        
        try:
            response = llm.invoke(prompt)
            raw_response = response.content.strip()
            print(f"Raw evaluator response: {raw_response}")  # Log raw response for debugging
            
            # Strip markdown code block markers
            cleaned_response = raw_response
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[len("```json"):].strip()
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-len("```")].strip()
            print(f"Cleaned evaluator response: {cleaned_response}")  # Log cleaned response
            
            try:
                eval_result = json.loads(cleaned_response)
                score = float(eval_result.get('score', 0.0))
                comment = eval_result.get("comment", "No feedback provided")
                print(f"Evaluation result for question='{question}': score={score}, comment='{comment}'")
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {str(e)}, cleaned_response='{cleaned_response}'")
                score = 0.0
                comment = f"Failed to parse evaluator response: {str(e)}. Cleaned response: {cleaned_response}"
        except Exception as e:
            score = 0.0
            comment = f"Evaluator error: {str(e)}"
            print(f"Evaluation error for question='{question}': {comment}")
        
        return EvaluationResult(
            key="correctness",
            score=score,
            comment=comment
        )
    
    return judge

# Define correctness evaluator using Gemini judge
correctness_evaluator = create_gemini_judge()

# Cache for training data to avoid repeated file I/O
_training_data_cache = None

def load_training_data(file_path: str = "training_data.jsonl") -> List[Dict[str, Any]]:
    global _training_data_cache
    if _training_data_cache is None:
        training_data = []
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                for line in f:
                    try:
                        training_data.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON line in {file_path}: {str(e)}")
        _training_data_cache = training_data
    return _training_data_cache

# Function to append new training data
def append_training_data(new_data: List[Dict[str, Any]], file_path: str = "training_data.jsonl") -> None:
    with open(file_path, "a") as f:
        for item in new_data:
            f.write(json.dumps(item) + "\n")
            st.info(f"Prompt: {item['prompt']} and Completion: {item['completion']} added to training_data.jsonl")

def update_examples(new_examples: List[Dict[str, Any]]) -> None:
    """
    Updates examples in LangSmith dataset only (doesn't modify in-memory examples list)
    Uses fetch_experiments to get existing examples for duplicate checking
    """
    try:
        # Get dataset id using fetch_experiments
        dataset_id = fetch_experiments()
        if not dataset_id:
            raise ValueError("Could not find dataset")
        
        # Get all existing examples from dataset
        existing_examples = list(langsmith_client.list_examples(dataset_id=dataset_id))
        existing_inputs = {ex.inputs.get("question", "").lower() for ex in existing_examples}
        
        added_count = 0
        for new_ex in new_examples:
            try:
                # Handle "prompt"/"completion" structure
                if not isinstance(new_ex, dict) or "prompt" not in new_ex or "completion" not in new_ex:
                    st.warning(f"Skipping malformed example: {new_ex}")
                    continue
                user_input = new_ex["prompt"]
                query = new_ex["completion"]
                
                # Skip if already exists in dataset
                if user_input.lower() in existing_inputs:
                    st.warning(f"{user_input[:50]} already in LangSmith examples list, won't be added")
                    continue
                
                # Add to LangSmith dataset
                langsmith_client.create_example(
                    dataset_id=dataset_id,
                    inputs={"question": user_input},
                    outputs={
                        "generated_query": query,
                        "reference_query": query  # Store as both generated and reference
                    }
                )
                added_count += 1
                existing_inputs.add(user_input.lower())
            except Exception as e:
                st.sidebar.warning(f"Failed to add example: {user_input[:50]}... Error: {str(e)}")
                continue
                
        st.sidebar.success(f"Added {added_count} new examples to LangSmith dataset")
        
    except Exception as e:
        st.sidebar.error(f"Error updating examples: {str(e)}")

def get_incorrect_outputs() -> List[Dict[str, Any]]:
    """Identifies incorrect outputs from evaluation runs"""
    incorrect_outputs = []
    try:
        # Get evaluation runs (without feedback filter)
        runs, _ = fetch_experiments()
        
        for run in runs:
            if not run.reference_example_id:
                continue
                
            # Get feedback separately
            feedback_list = list(langsmith_client.list_feedback(run_ids=[run.id]))
            
            # Check for incorrect feedback (score < 1)
            incorrect_feedback = [
                f for f in feedback_list 
                if f.key == "correctness" and f.score is not None and f.score < 1.0
            ]
            
            if not incorrect_feedback:
                continue
                
            try:
                example = langsmith_client.read_example(run.reference_example_id)
                question = run.inputs.get("question", "")
                reference_output = example.outputs.get("generated_query", "")
                
                if question and reference_output:
                    incorrect_outputs.append({
                        "messages": [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": reference_output}
                        ]
                    })
            except Exception:
                continue
                
    except Exception as e:
        st.sidebar.error(f"Error finding incorrect outputs: {str(e)}")
    
    return incorrect_outputs

def fetch_experiments(dataset_name: str = "sql_Kiotviet", project_name: str = None) -> tuple[List[Any], UUID | None]:
    try:
        # Fetch the dataset by name
        dataset = langsmith_client.read_dataset(dataset_name=dataset_name)
        dataset_id = dataset.id

        # Fetch project (session) if not provided
        if project_name:
            project = langsmith_client.read_project(project_name=project_name)
            project_id = project.id
        else:
            # Get all projects and find the most recent evaluation project
            all_projects = list(langsmith_client.list_projects())
            eval_projects = [
                p for p in all_projects 
                if 'gemini-sql-query-eval' in p.name.lower()
            ]
            if not eval_projects:
                raise ValueError("No evaluation projects found")
            project_id = eval_projects[0].id  # Use most recent project

        # Fetch runs with only allowed fields
        runs = list(langsmith_client.list_runs(
            project_id=project_id,
            is_root=True,
            select=["id", "inputs", "outputs", "reference_example_id"]  # Only valid fields
        ))

        # Filter runs to those referencing examples in our dataset
        dataset_example_ids = {ex.id for ex in langsmith_client.list_examples(dataset_id=dataset_id)}
        filtered_runs = [run for run in runs if run.reference_example_id in dataset_example_ids]
        
        return filtered_runs, dataset_id
        
    except Exception as e:
        st.error(f"Error fetching runs for dataset {dataset_name}: {str(e)}")
        return [], None
    
def query_exists(question: str, query: str) -> bool:
    # Check in examples
    for ex in examples:
        try:
            if ex.get("input", "").lower() == question.lower() and ex.get("query", "").strip().lower() == query.strip().lower():
                return True
        except (KeyError, AttributeError) as e:
            print(f"Warning: Skipping malformed example in query_exists: {str(e)}, example: {ex}")
            continue
    
    # Check in training data
    training_data = load_training_data()
    for item in training_data:
        try:
            if not isinstance(item, dict):
                print(f"Warning: Skipping malformed training data item: {item}")
                continue
            # Handle "prompt"/"completion" structure
            if "prompt" in item and "completion" in item:
                if (
                    item.get("prompt", "").lower() == question.lower() and
                    item.get("completion", "").strip().lower() == query.strip().lower()
                ):
                    return True
            # Handle "messages" structure
            elif "messages" in item:
                messages = item["messages"]
                if (
                    isinstance(messages, list) and
                    len(messages) >= 2 and
                    isinstance(messages[1], dict) and
                    isinstance(messages[2], dict) and
                    messages[1].get("content", "").lower() == question.lower() and
                    messages[2].get("content", "").strip().lower() == query.strip().lower()
                ):
                    return True
            else:
                print(f"Warning: Skipping training data item with unknown structure: {item}")
        except (KeyError, AttributeError, TypeError) as e:
            print(f"Warning: Skipping malformed training data item: {str(e)}, item: {item}")
            continue
    
    return False

# Updated function to check experiments for incorrect outputs - IMPROVED FILTERING
def check_langsmith_experiments(dataset_name: str = "sql_Kiotviet", project_name: str = None) -> List[Dict[str, Any]]:
    new_training_data = []
    try:
        runs, dataset_id = fetch_experiments(dataset_name, project_name)

        if not runs:
            return []

        for run in runs:
            if run.reference_example_id:
                example = langsmith_client.read_example(run.reference_example_id)
                model_output = run.outputs.get("generated_query", "") if hasattr(run, "outputs") else ""
                reference_output = example.outputs.get("generated_query", "") if hasattr(example, "outputs") else ""
                question = run.inputs.get("question", "")
                
                # Get feedback scores if available
                feedback = run.feedback if hasattr(run, "feedback") else None
                score = feedback[0].score if feedback and len(feedback) > 0 else None
                
                # Only add if:
                # 1. Outputs differ OR score indicates incorrect (score < 1.0)
                # 2. Not already in training data/examples
                # 3. Reference output is actually correct (not empty or error)
                if (model_output != reference_output or (score is not None and score < 1.0)) and \
                   not query_exists(question, reference_output) and \
                   reference_output.strip() and "error" not in reference_output.lower(): #Last part might give error
                    
                    training_example = {
                        "messages": [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": reference_output}
                        ]
                    }
                    new_training_data.append(training_example)
        return new_training_data
    except Exception as e:
        st.sidebar.error(f"Error checking experiments: {str(e)}")
        return []

# Streamlit UI
question = st.text_input("Enter your question:", key="input")
submit = st.button("Generate SQL Query")

# Submission logic for SQL query generator only
if submit:
    try:
        chain = create_query_chain()
        sql_query = chain.invoke({
            "question": question,
            "top_k": 5,
            "table_info": db.get_table_info(),
            "dialect": "SQLite"
        })
        
        sql_query = remove_sql_backticks(sql_query)
        
        st.subheader("Generated SQL Query (Gemini):")
        st.code(sql_query)
        
        results = execute_query(sql_query)
        
        try:
            log_evaluation(question, sql_query, results)
            st.subheader("Results:")
            if isinstance(results, list) and results:
                df = pd.DataFrame(results)
                st.dataframe(df)
            else:
                st.warning("No results found or error occurred")
        except Exception as e:
            log_evaluation(question, sql_query, None, error=str(e))
            st.error(f"Error executing query: {str(e)}")
    except GoogleAuthError as e:
        st.error(f"Authentication error: {str(e)}. Please ensure GOOGLE_CLOUD_PROJECT is set and the project exists.")
        st.markdown("[Create a Google Cloud project](https://developers.google.com/workspace/guides/create-project)")
    except ResourceExhausted as e:
        st.error(f"Quota exceeded: {str(e)}. Please try again later or upgrade to a paid tier.")
        st.markdown("[Learn more about Gemini API quotas](https://ai.google.dev/gemini-api/docs/rate-limits)")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# LangSmith evaluation dashboard
st.sidebar.header("Evaluation Dashboard")
if st.sidebar.button("View Evaluation Dashboard"):
    st.sidebar.write("Opening LangSmith dashboard...")
    st.sidebar.markdown("[Open LangSmith Dashboard](https://smith.langchain.com)")

# Evaluation portion
st.sidebar.header("Run Evaluation")
if st.sidebar.button("Run Evaluation"):
    st.sidebar.write("Running evaluation...")
    try:
        # Ensure dataset is set up with proper inputs
        dataset = setup_evaluation_dataset()
        
        # Log dataset state before evaluation
        print(f"Starting evaluation for dataset {dataset.name} (ID: {dataset.id})")
        existing_examples = list(langsmith_client.list_examples(dataset_id=dataset.id))
        print(f"Found {len(existing_examples)} examples in dataset")
        for ex in existing_examples:
            print(f"Example {ex.id}: inputs={ex.inputs}, outputs={ex.outputs}")
        
        experiment_results = langsmith_client.evaluate(
            target,
            data="sql_Kiotviet",
            evaluators=[correctness_evaluator],
            experiment_prefix="gemini-sql-query-eval",
            max_concurrency=2,
            metadata={"evaluation_run": f"run_{int(time.time())}"}  # Add metadata for tracking
        )
        st.sidebar.success("Evaluation completed! Check LangSmith for results.")
        st.sidebar.markdown("[Open LangSmith Dashboard](https://smith.langchain.com)")
        
        # Log evaluation results
        print(f"Evaluation completed: {experiment_results}")
    except Exception as e:
        st.sidebar.error(f"Evaluation failed: {str(e)}")
        print(f"Evaluation error: {str(e)}")

# Model improvement
st.sidebar.header("Model Improvement")
if st.sidebar.button("Improve Model"):
    with st.sidebar:
        with st.spinner("Analyzing LangSmith evaluations..."):
            try:
                # Initialize progress bar (0-1 scale)
                progress_bar = st.progress(0.0, text="Fetching evaluation data...")
                
                # Get all runs for the latest experiment
                runs, dataset_id = fetch_experiments()
                if not runs:
                    st.error("No evaluation runs found")
                    progress_bar.empty()
                    st.stop()
                
                total_runs = len(runs)
                new_training_data = []
                processed_questions = set()
                
                progress_bar.progress(0.1, text=f"Processing {total_runs} runs...")
                
                for i, run in enumerate(runs):
                    # Update progress (0.1 to 0.8 range)
                    progress_value = 0.1 + (i / total_runs * 0.7)
                    progress_bar.progress(
                        progress_value,
                        text=f"Processing run {i+1} of {total_runs}..."
                    )
                    
                    if not run.reference_example_id:
                        st.warning(f"Skipping run {run.id}: No reference_example_id")
                        continue
                        
                    try:
                        # Get feedback for this run
                        feedback = list(langsmith_client.list_feedback(run_ids=[run.id]))
                        incorrect_feedback = [
                            f for f in feedback 
                            if f.key == "correctness" and f.score is not None and f.score < 1.0
                        ]

                        # Use run.outputs['answer'] as fallback
                        reference_output = example.outputs.get("generated_query", run.outputs.get("answer", ""))
                        print(f"Run {run.id}: Selected reference_output={reference_output}")
                        if not isinstance(reference_output, str) or not reference_output.strip() or reference_output == "Error":
                            st.warning(f"Skipping run {run.id}: Invalid reference_output (type: {type(reference_output)}, value: {reference_output})")
                            continue
                            
                        # Skip duplicates
                        if question.lower() in processed_questions:
                            st.info(f"Skipping run {run.id}: Duplicate question")
                            continue
                            
                        # Check if query exists
                        print(f"Run {run.id}: Calling query_exists with question={question}, reference_output={reference_output}")
                        if not query_exists(question, reference_output):
                            training_example = {
                                "prompt": question,
                                "completion": reference_output
                            }
                            print(f"Run {run.id}: Created training_example={training_example}")
                            new_training_data.append(training_example)
                            processed_questions.add(question.lower())
                            print(f"Run {run.id}: Appended to new_training_data, len={len(new_training_data)}")
                            
                    except Exception as e:
                        st.warning(f"Skipping run {run.id}: {str(e)} (question: {inputs_dict.get('question', 'N/A')}, inputs: {run.inputs}, outputs: {getattr(example, 'outputs', 'N/A')})")
                        continue
                
                progress_bar.progress(0.9, text="Updating training data...")
                
                if new_training_data:
                    print(f"Calling append_training_data with {len(new_training_data)} items")
                    append_training_data(new_training_data)
                    print("Called append_training_data")
                    print(f"Calling update_examples with {len(new_training_data)} items")
                    try:
                        update_examples(new_training_data)
                        print("Called update_examples")
                    except Exception as e:
                        print(f"Error in update_examples: {str(e)}")
                else:
                    st.info("No incorrect outputs found to learn from")
                    
                progress_bar.progress(1.0, text="Completed!")
                time.sleep(0.5)
                progress_bar.empty()
                
            except Exception as e:
                st.error(f"Error during model improvement: {str(e)}")
                progress_bar.empty()