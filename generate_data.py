import argparse
import json
import sys
import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm

# --- Constants ---
OLLAMA_API_ENDPOINT = "/api/chat"

# --- System Prompt ---
# This prompt guides the model to generate data in the desired JSON format.
SYSTEM_PROMPT = """
You are an expert synthetic data generator. Your task is to generate high-quality, instruction-style input/output pairs for fine-tuning LLMs.
You will be given a few examples of existing pairs. Based on these examples, you must generate a new, unique set of pairs that follow the same pattern and style.

RULES:
1.  GENERATE ONLY a valid JSON object containing a single key "data".
2.  The "data" key must hold a list of JSON objects.
3.  Each object in the list must have two keys: "input" and "output".
4.  The content for "input" and "output" should be text, mirroring the style of the examples provided.
5.  DO NOT include any other text, explanations, or apologies before or after the JSON object.
"""

def check_ollama_connection(host: str) -> None:
    """Checks if the Ollama server is running and accessible."""
    print(f"Checking connection to Ollama at {host}...")
    try:
        response = requests.get(host)
        response.raise_for_status()
        print("✅ Ollama server is running.")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: Could not connect to Ollama at {host}.")
        print("Please ensure Ollama is running.")
        print(f"Details: {e}")
        sys.exit(1)

def load_initial_data(seed_file: Path, output_file: Path, input_col: str, output_col: str) -> tuple[pd.DataFrame, set]:
    """Loads the seed data and any previously generated data."""
    print(f"Loading seed data from: {seed_file}")
    if not seed_file.exists():
        print(f"❌ Error: Seed file not found at {seed_file}")
        sys.exit(1)

    # Load seed data
    try:
        seed_df = pd.read_csv(seed_file)
        # Validate required columns
        if input_col not in seed_df.columns or output_col not in seed_df.columns:
            print(f"❌ Error: Seed file must contain '{input_col}' and '{output_col}' columns.")
            sys.exit(1)
        # Standardize column names for internal use
        seed_df = seed_df.rename(columns={input_col: 'input', output_col: 'output'})
        seed_df = seed_df[['input', 'output']]
    except Exception as e:
        print(f"❌ Error reading seed file: {e}")
        sys.exit(1)

    # Load existing output data if it exists
    if output_file.exists():
        print(f"Found existing output file. Loading to continue generation: {output_file}")
        try:
            output_df = pd.read_csv(output_file)
            if 'input' not in output_df.columns or 'output' not in output_df.columns:
                 print(f"⚠️ Warning: Output file exists but doesn't have 'input'/'output' columns. It will be overwritten.")
                 combined_df = seed_df
            else:
                 combined_df = pd.concat([seed_df, output_df[['input', 'output']]], ignore_index=True)
        except Exception as e:
            print(f"⚠️ Warning: Could not read existing output file. It may be corrupted or empty. Will start fresh. Error: {e}")
            combined_df = seed_df
    else:
        print("No existing output file found. Starting fresh.")
        combined_df = seed_df

    # Create a set of existing inputs for quick duplicate checks
    existing_inputs = set(combined_df['input'].astype(str))
    print(f"Loaded {len(seed_df)} seed examples and {len(combined_df) - len(seed_df)} existing examples.")
    print(f"Total unique examples to sample from: {len(existing_inputs)}")
    
    return combined_df, existing_inputs


def generate_prompt(sample_df: pd.DataFrame, user_prompt: str = None) -> str:
    """Creates the user prompt with few-shot examples, optionally prepending a user description."""
    prompt_parts = []
    if user_prompt:
        prompt_parts.append(user_prompt.strip() + "\n")
    examples = []
    for _, row in sample_df.iterrows():
        examples.append(json.dumps({'input': row['input'], 'output': row['output']}))
    examples_str = ",\n".join(examples)
    prompt_parts.append(f"Here are some examples:\n[\n{examples_str}\n]\n\nNow, generate 2 new, unique input/output pairs following this format.")
    return "\n".join(prompt_parts)

def call_ollama_api(host: str, model: str, system_prompt: str, user_prompt: str, temperature: float) -> list[dict]:
    """Calls the Ollama chat API and parses the JSON response."""
    url = f"{host}{OLLAMA_API_ENDPOINT}"
    payload = {
        "model": model,
        "format": "json",
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "options": {
            "temperature": temperature
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        response_json = response.json()
        content_str = response_json.get('message', {}).get('content', '')
        
        if not content_str:
            print("\n⚠️ Warning: Received empty content from model.")
            return []
            
        # The model should return a JSON object string, which we parse
        data = json.loads(content_str)
        
        # Validate the received data structure
        if 'data' in data and isinstance(data['data'], list):
            return data['data']
        else:
            print(f"\n⚠️ Warning: Model output did not match expected format. Received: {data}")
            return []

    except requests.exceptions.Timeout:
        print("\n❌ Error: Request to Ollama timed out. The model might be too large for your machine or is taking too long to respond.")
        return []
    except requests.exceptions.RequestException as e:
        print(f"\n❌ API Error: {e}")
        return []
    except json.JSONDecodeError:
        print(f"\n❌ JSON Decode Error: Model did not return valid JSON. Response content: {content_str}")
        return []
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        return []

def main():
    """Main function to run the data generation script."""
    parser = argparse.ArgumentParser(description="Generate synthetic data using Ollama.")
    parser.add_argument("-s", "--seed-file", type=Path, required=True, help="Path to the input seed CSV file.")
    parser.add_argument("-o", "--output-file", type=Path, required=True, help="Path to the output CSV file.")
    parser.add_argument("-m", "--model", type=str, required=True, help="Name of the Ollama model to use (e.g., 'llama3:instruct').")
    parser.add_argument("-n", "--num-to-generate", type=int, default=100, help="Total number of synthetic data points to generate.")
    parser.add_argument("-k", "--num-samples", type=int, default=3, help="Number of few-shot examples for each prompt.")
    parser.add_argument("-t", "--temperature", type=float, default=0.7, help="Generation temperature for the model.")
    parser.add_argument("-i", "--input-column", type=str, default="input", help="Name of the 'input' column in the seed file.")
    parser.add_argument("-p", "--output-column", type=str, default="output", help="Name of the 'output' column in the seed file.")
    parser.add_argument("-H", "--ollama-host", type=str, default="http://localhost:11434", help="Host address for the Ollama API.")
    parser.add_argument("--user-prompt", type=str, default=None, help="Optional description to guide the type of synthetic data to generate.")
    args = parser.parse_args()

    # --- Initial Setup ---
    check_ollama_connection(args.ollama_host)
    all_data_df, existing_inputs = load_initial_data(args.seed_file, args.output_file, args.input_column, args.output_column)
    
    # --- Generation Loop ---
    new_data_list = []

    # Store original values for dynamic adjustment
    original_temperature = args.temperature
    original_num_samples = args.num_samples
    current_temperature = original_temperature
    current_num_samples = original_num_samples
    max_temperature = 1.5
    max_num_samples = 40

    # Calculate how many data points are already in the output file
    if args.output_file.exists():
        try:
            existing_output_count = len(pd.read_csv(args.output_file))
        except pd.errors.EmptyDataError:
             existing_output_count = 0
    else:
        existing_output_count = 0

    num_needed = args.num_to_generate - existing_output_count
    
    if num_needed <= 0:
        print(f"✅ Output file already contains {existing_output_count} / {args.num_to_generate} desired records. Nothing to do.")
        sys.exit(0)
    
    print(f"\nStarting generation of {num_needed} new data points...")
    pbar = tqdm(total=num_needed, desc="Generating Data")

    while len(new_data_list) < num_needed:
        # 1. Sample data for the prompt
        sample_df = all_data_df.sample(n=min(current_num_samples, len(all_data_df)))
        
        # 2. Create the prompt
        user_prompt = generate_prompt(sample_df, args.user_prompt)
        
        # 3. Call the API
        generated_pairs = call_ollama_api(
            host=args.ollama_host,
            model=args.model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=current_temperature
        )
        
        # 4. Process and save results
        new_unique_found = False
        if generated_pairs:
            for pair in generated_pairs:
                if isinstance(pair, dict) and 'input' in pair and 'output' in pair:
                    # Check for duplicates
                    if str(pair['input']) not in existing_inputs:
                        new_data_list.append(pair)
                        existing_inputs.add(str(pair['input']))
                        new_unique_found = True
                        
                        # Append to master dataframe for future sampling
                        all_data_df = pd.concat([all_data_df, pd.DataFrame([pair])], ignore_index=True)

                        # Append to CSV file immediately for resilience
                        temp_df = pd.DataFrame([pair])
                        temp_df.to_csv(
                            args.output_file, 
                            mode='a', 
                            header=not args.output_file.exists(), 
                            index=False
                        )
                        pbar.update(1)

                        if len(new_data_list) >= num_needed:
                            break
                else:
                    print(f"\n⚠️ Skipping malformed pair: {pair}")
        if new_unique_found:
            # Reset to original values after a successful unique generation
            current_temperature = original_temperature
            current_num_samples = original_num_samples
        else:
            # If all were repeats, increase temp and num_samples for next round
            current_temperature = min(current_temperature + 0.05, max_temperature)
            current_num_samples = min(current_num_samples + 5, max_num_samples)
            print(f"No new unique pairs found. Increasing temperature to {current_temperature} and num_samples to {current_num_samples} for next attempt.")
        if len(new_data_list) >= num_needed:
            break

    pbar.close()
    print(f"\n✅ Generation complete. Total of {len(new_data_list)} new records saved to {args.output_file}")


if __name__ == "__main__":
    main()