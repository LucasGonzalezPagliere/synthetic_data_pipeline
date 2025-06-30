# Synthetic Data Generator for LLM Fine-Tuning

This project provides a command-line tool to generate high-quality, instruction-style input/output pairs for fine-tuning large language models (LLMs) using the [Ollama](https://ollama.com/) API. It leverages few-shot prompting and adaptive generation strategies to efficiently create diverse synthetic datasets.

## Features
- Few-shot prompt generation using your own seed data
- Adaptive temperature and sample size to avoid repeated outputs
- Supports resuming from existing output files
- Customizable prompt to guide the type of data generated
- Progress bar and robust error handling

## Installation
1. **Clone this repository**
2. **Install dependencies** (Python 3.8+ recommended):
   ```sh
   pip install -r requirements.txt
   ```
3. **Install and run Ollama** ([see Ollama docs](https://ollama.com/)) and ensure your desired model is available and running.

## Usage
Run the script from the command line:
```sh
python generate_data.py \
  -s rude_responses_seed.csv \
  -o rude_synthetic_data.csv \
  -m llama3:instruct \
  -n 100 \
  -k 3 \
  -t 0.7 \
  --user-prompt "Generate polite customer service responses for technical support scenarios."
```

### Arguments
- `-s`, `--seed-file` (required): Path to your input seed CSV file (must have input/output columns)
- `-o`, `--output-file` (required): Path to the output CSV file
- `-m`, `--model` (required): Name of the Ollama model to use (e.g., `llama3:instruct`)
- `-n`, `--num-to-generate`: Total number of synthetic data points to generate (default: 100)
- `-k`, `--num-samples`: Number of few-shot examples for each prompt (default: 3)
- `-t`, `--temperature`: Generation temperature for the model (default: 0.7)
- `-i`, `--input-column`: Name of the 'input' column in the seed file (default: `input`)
- `-p`, `--output-column`: Name of the 'output' column in the seed file (default: `output`)
- `-H`, `--ollama-host`: Host address for the Ollama API (default: `http://localhost:11434`)
- `--user-prompt`: Optional description to guide the type of synthetic data to generate

## Preparing Your Seed Data
- Your seed CSV should have at least two columns: one for input, one for output (default names: `input`, `output`).
- Example:
  | input                  | output                  |
  |------------------------|-------------------------|
  | How are you?           | I'm fine, thank you!    |
  | What's your name?      | I'm an AI assistant.    |

## Ollama Requirements
- You must have [Ollama](https://ollama.com/) installed and running locally or on your specified host.
- The model you specify (e.g., `llama3:instruct`) must be available in your Ollama instance.

## Tips for Best Results
- Use a diverse and representative seed file for better synthetic data quality.
- If generation slows down, the script will automatically increase temperature and sample size to avoid repeats.
- You can resume interrupted runs; the script will not duplicate existing outputs.
- Use the `--user-prompt` argument to further guide the style or domain of generated data.

## License
[Your License Here] 