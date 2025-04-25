# Open-Ended Question Analyzer

A Python script to identify and analyze open-ended questions in text documents using multiple LLM APIs (OpenAI, Claude, Gemini, and DeepSeek) and provide consensus analysis.

## Overview

This tool helps researchers and content analysts identify open-ended questions in text by leveraging multiple large language models. It then analyzes the consensus and differences between model outputs to provide insights into:

- Which questions are consistently identified across all models
- Questions uniquely identified by specific models
- Semantic similarity between model outputs
- Performance metrics for each model

## Features

- Processes text files to identify open-ended questions
- Uses four major LLM APIs:
  - OpenAI (GPT-3.5 Turbo or GPT-4)
  - Anthropic's Claude
  - Google's Gemini
  - DeepSeek
- Performs consensus analysis to identify agreement between models
- Calculates semantic similarity between identified questions
- Measures performance metrics (processing time, confidence scores)
- Exports all results as structured JSON for further analysis

## Requirements

- Python 3.8+
- API keys for the LLM services

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys (see `.env.example`)

## Usage

Basic usage:

```bash
python open_ended_question_analyzer.py input_file.txt
```

Specify an output file:

```bash
python open_ended_question_analyzer.py input_file.txt -o results.json
```

## Output Format

The script generates a JSON file with the following structure:

```json
{
  "file": "input_file.txt",
  "models": {
    "openai": {
      "questions": [...],
      "metadata": {...}
    },
    "claude": {...},
    "gemini": {...},
    "deepseek": {...}
  },
  "consensus_analysis": {
    "all_models": [...],
    "majority": [...],
    "unique": [...],
    "unique_to_openai": [...],
    "unique_to_claude": [...],
    "unique_to_gemini": [...],
    "unique_to_deepseek": [...]
  },
  "performance_metrics": {...},
  "semantic_similarity": {...}
}
```

## Example

Using the included sample file:

```bash
python open_ended_question_analyzer.py sample.txt
```

## Notes

- Ensure your text file isn't too large, as API rate limits and token limits may apply.
- The semantic similarity analysis uses TF-IDF and cosine similarity as a basic approach.
- API costs may be incurred when using this tool, depending on your API usage plans.

## License

MIT 