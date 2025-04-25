# Open-Ended Question Analyzer

A Python tool that identifies and analyzes open-ended questions in text documents using multiple LLM APIs (OpenAI, Claude, Gemini, and DeepSeek) and provides consensus analysis with visualizations.

## Overview

This tool helps researchers and content analysts identify open-ended questions in text by leveraging multiple large language models. It normalizes and compares the outputs to provide insights into:

- Which questions are consistently identified across all models
- Questions uniquely identified by specific models
- Patterns in how different models interpret open-ended questions
- Confidence scores for each identified question

## Features

### Core Analysis
- Processes text files to identify open-ended questions
- Uses four major LLM APIs:
  - OpenAI (GPT-4o-mini or other models)
  - Anthropic's Claude
  - Google's Gemini
  - DeepSeek
- Performs robust question normalization and similarity comparison
- Handles slight variations in how models identify the same questions
- Adapts to available API keys (only uses APIs you have keys for)

### Results and Visualizations
- Provides clear terminal output summarizing findings
- Generates comprehensive JSON output with detailed analysis
- Automatically creates visualizations including:
  - Question counts per model
  - Consensus analysis charts
  - Confidence score comparisons
- Produces an HTML report with all visualizations and tables
- Automatically opens the report in your default browser

### User Interface
- Simple command-line interface
- Command-line arguments for customization:
  - `--no-visualize` to skip visualization
  - `--no-browser` to prevent browser opening
  - `-o/--output` to specify output file location

## Requirements

- Python 3.8+
- API keys for at least one of the LLM services

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   GOOGLE_API_KEY=your_google_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   DEEPSEEK_API_KEY=your_deepseek_api_key
   ```
   (Only add the API keys you have available)

## Usage

### Basic Usage

Run analysis on a text file:

```bash
python open_ended_question_analyzer.py sample.txt
```

This will:
1. Analyze the text using available LLM APIs
2. Save the analysis as `sample.analysis.json`
3. Generate visualizations in a `visualizations` directory
4. Open an HTML report in your default browser

### Additional Options

Skip visualization step:
```bash
python open_ended_question_analyzer.py sample.txt --no-visualize
```

Don't open the browser automatically:
```bash
python open_ended_question_analyzer.py sample.txt --no-browser
```

Specify a custom output file:
```bash
python open_ended_question_analyzer.py sample.txt -o results/my_analysis.json
```

### Running Visualizations Separately

You can also generate visualizations for an existing analysis file:
```bash
python visualize_results.py sample.analysis.json
```

## How It Works

1. **Text Processing**: The input file is read and processed
2. **LLM Analysis**: The text is sent to multiple LLM APIs, asking them to identify open-ended questions
3. **Question Normalization**: Identified questions are normalized to handle formatting differences
4. **Similarity Detection**: Questions with minor variations are grouped together
5. **Consensus Analysis**: The tool determines which questions were found by multiple models
6. **Visualization**: Results are visualized and presented in an HTML report

## Output Format

The JSON output includes:
- All identified questions with their confidence scores
- Consensus analysis (questions agreed upon by multiple models)
- Variants of similar questions identified by different models
- Performance metrics for each model

## Example

The repository includes a sample.txt file with examples of open-ended questions for testing.

## Notes

- Ensure your text file isn't too large, as API rate limits and token limits may apply
- API costs may be incurred when using this tool, depending on your API usage plans
- Only the APIs you have keys for will be used in the analysis