#!/usr/bin/env python3
"""
Open-Ended Question Analyzer

This script processes a text file and uses multiple LLM APIs to identify 
open-ended questions, then analyzes the consensus and differences between models.
"""

import os
import json
import time
import argparse
import subprocess
import re
import webbrowser
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Import LLM API clients
import openai
from anthropic import Anthropic
import google.generativeai as genai
import requests

# For analysis
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load environment variables from .env file
load_dotenv()

# Configure API clients
openai.api_key = os.getenv("OPENAI_API_KEY")
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

class OpenEndedQuestionAnalyzer:
    """Analyzes text to identify open-ended questions using multiple LLM APIs."""
    
    def __init__(self, input_file: str, output_file: Optional[str] = None):
        """
        Initialize the analyzer with input and output file paths.
        
        Args:
            input_file: Path to the text file to analyze
            output_file: Path where to save the results (JSON)
        """
        self.input_file = Path(input_file)
        self.output_file = Path(output_file) if output_file else Path(input_file).with_suffix('.analysis.json')
        self.text_content = self._read_file()
        self.results = {
            "file": self.input_file.name,
            "models": {},
            "consensus_analysis": {},
            "performance_metrics": {}
        }
    
    def _read_file(self) -> str:
        """Read the content of the input file."""
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise Exception(f"Error reading input file: {e}")
    
    def analyze_with_openai(self) -> Dict[str, Any]:
        """Query OpenAI API to identify open-ended questions."""
        start_time = time.time()
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",  # or "gpt-4" if needed
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that identifies open-ended questions in text."},
                    {"role": "user", "content": f"""
                    Analyze the following text and identify all open-ended questions.
                    Return your analysis as a JSON object with the following structure:
                    {{
                        "open_ended_questions": [
                            {{
                                "question": "The full text of the question",
                                "confidence": 0.95, // Your confidence score between 0 and 1
                                "context": "Surrounding context if relevant"
                            }}
                        ]
                    }}
                    
                    Text to analyze:
                    {self.text_content}
                    """
                    }
                ]
            )
            
            result = json.loads(response.choices[0].message.content)
            end_time = time.time()
            
            return {
                "questions": result.get("open_ended_questions", []),
                "metadata": {
                    "model": response.model,
                    "processing_time": end_time - start_time
                }
            }
            
        except Exception as e:
            return {
                "questions": [],
                "metadata": {
                    "error": str(e)
                }
            }
    
    def analyze_with_claude(self) -> Dict[str, Any]:
        """Query Claude API to identify open-ended questions."""
        start_time = time.time()
        
        try:
            response = anthropic_client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=4000,
                system="You are a helpful assistant that identifies open-ended questions in text.",
                messages=[
                    {
                        "role": "user", 
                        "content": f"""
                        Analyze the following text and identify all open-ended questions.
                        Return your analysis as a JSON object with the following structure:
                        {{
                            "open_ended_questions": [
                                {{
                                    "question": "The full text of the question",
                                    "confidence": 0.95, // Your confidence score between 0 and 1
                                    "context": "Surrounding context if relevant"
                                }}
                            ]
                        }}
                        
                        Text to analyze:
                        {self.text_content}
                        """
                    }
                ]
            )
            
            # Extract JSON from the response
            content = response.content[0].text
            # Find the JSON part in the response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            json_str = content[json_start:json_end]
            
            result = json.loads(json_str)
            end_time = time.time()
            
            return {
                "questions": result.get("open_ended_questions", []),
                "metadata": {
                    "model": response.model,
                    "processing_time": end_time - start_time
                }
            }
            
        except Exception as e:
            return {
                "questions": [],
                "metadata": {
                    "error": str(e)
                }
            }
    
    def analyze_with_gemini(self) -> Dict[str, Any]:
        """Query Google's Gemini API to identify open-ended questions."""
        start_time = time.time()
        
        try:
            model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')
            
            prompt = f"""
            Analyze the following text and identify all open-ended questions.
            Return your analysis as a JSON object with the following structure:
            {{
                "open_ended_questions": [
                    {{
                        "question": "The full text of the question",
                        "confidence": 0.95, // Your confidence score between 0 and 1
                        "context": "Surrounding context if relevant"
                    }}
                ]
            }}
            
            Text to analyze:
            {self.text_content}
            """
            
            response = model.generate_content(prompt)
            
            # Extract JSON from the response
            content = response.text
            # Find the JSON part in the response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            json_str = content[json_start:json_end]
            
            result = json.loads(json_str)
            end_time = time.time()
            
            return {
                "questions": result.get("open_ended_questions", []),
                "metadata": {
                    "model": "gemini-pro",
                    "processing_time": end_time - start_time
                }
            }
            
        except Exception as e:
            return {
                "questions": [],
                "metadata": {
                    "error": str(e)
                }
            }
    
    def analyze_with_deepseek(self) -> Dict[str, Any]:
        """Query DeepSeek API to identify open-ended questions."""
        start_time = time.time()
        
        # DeepSeek API endpoint
        endpoint = "https://api.deepseek.com/v1/chat/completions"
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
            }
            
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that identifies open-ended questions in text."},
                    {"role": "user", "content": f"""
                    Analyze the following text and identify all open-ended questions.
                    Return your analysis as a JSON object with the following structure:
                    {{
                        "open_ended_questions": [
                            {{
                                "question": "The full text of the question",
                                "confidence": 0.95, // Your confidence score between 0 and 1
                                "context": "Surrounding context if relevant"
                            }}
                        ]
                    }}
                    
                    Text to analyze:
                    {self.text_content}
                    """
                    }
                ]
            }
            
            response = requests.post(endpoint, headers=headers, json=data)
            response.raise_for_status()
            response_data = response.json()
            
            content = response_data["choices"][0]["message"]["content"]
            # Find the JSON part in the response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            json_str = content[json_start:json_end]
            
            result = json.loads(json_str)
            end_time = time.time()
            
            return {
                "questions": result.get("open_ended_questions", []),
                "metadata": {
                    "model": "deepseek-chat",
                    "processing_time": end_time - start_time
                }
            }
            
        except Exception as e:
            return {
                "questions": [],
                "metadata": {
                    "error": str(e)
                }
            }
    
    def run_analysis(self) -> None:
        """Run the complete analysis using all LLM APIs."""
        print(f"Analyzing text from: {self.input_file}")
        
        # Only run models with valid API keys
        if os.getenv("OPENAI_API_KEY"):
            self.results["models"]["openai"] = self.analyze_with_openai()
            print("OpenAI analysis complete")
        else:
            print("Skipping OpenAI analysis (no API key)")
        
        if os.getenv("ANTHROPIC_API_KEY"):
            self.results["models"]["claude"] = self.analyze_with_claude()
            print("Claude analysis complete")
        else:
            print("Skipping Claude analysis (no API key)")
        
        if os.getenv("GOOGLE_API_KEY"):
            self.results["models"]["gemini"] = self.analyze_with_gemini()
            print("Gemini analysis complete")
        else:
            print("Skipping Gemini analysis (no API key)")
        
        if os.getenv("DEEPSEEK_API_KEY"):
            self.results["models"]["deepseek"] = self.analyze_with_deepseek()
            print("DeepSeek analysis complete")
        else:
            print("Skipping DeepSeek analysis (no API key)")
        
        # Ensure we have at least one model with results
        if not self.results["models"]:
            print("Error: No API keys provided. Please add at least one API key to .env file.")
            return
        
        # Perform consensus analysis
        self._analyze_consensus()
        print("Consensus analysis complete")
        
        # Calculate performance metrics
        self._calculate_performance_metrics()
        print("Performance metrics calculated")
        
        # Calculate semantic similarity if applicable
        multiple_models = len(self.results["models"]) > 1
        
        # Check if any model returned zero questions
        has_empty_results = False
        models_with_empty_results = []
        
        for model_name, model_data in self.results["models"].items():
            if len(model_data["questions"]) == 0:
                has_empty_results = True
                models_with_empty_results.append(model_name)
        
        if multiple_models and not has_empty_results:
            self._calculate_semantic_similarity()
            print("Semantic similarity calculated")
        elif not multiple_models:
            print("Skipping semantic similarity (need at least 2 models)")
            self.results["semantic_similarity"] = {"error": "Need at least 2 models for similarity comparison"}
        else:
            empty_models = ", ".join(models_with_empty_results)
            print(f"Skipping semantic similarity (no questions returned by: {empty_models})")
            self.results["semantic_similarity"] = {"error": f"No questions returned by: {empty_models}"}
        
        # Save results
        self._save_results()
        print(f"Results saved to: {self.output_file}")
        
        # Print summary to terminal
        self._print_terminal_summary()
    
    def _print_terminal_summary(self) -> None:
        """Print a human-readable summary of the analysis to the terminal."""
        print("\n" + "="*80)
        print(f"SUMMARY ANALYSIS FOR: {self.input_file.name}")
        print("="*80)
        
        # Model counts
        print("\nQUESTIONS IDENTIFIED PER MODEL:")
        print("-"*50)
        for model_name, model_data in self.results["models"].items():
            questions_count = len(model_data["questions"])
            model_display = model_name.upper()
            print(f"{model_display}: {questions_count} questions")
        
        # Consensus information
        consensus = self.results["consensus_analysis"]
        print("\nCONSENSUS ANALYSIS:")
        print("-"*50)
        print(f"Questions found by ALL models: {len(consensus['all_models'])}")
        print(f"Questions found by MULTIPLE (but not all) models: {len(consensus['majority'])}")
        print(f"Questions found by only ONE model: {len(consensus['unique'])}")
        
        # Show questions agreed by all models
        if consensus['all_models']:
            print("\nQUESTIONS IDENTIFIED BY ALL MODELS:")
            print("-"*50)
            for i, question in enumerate(consensus['all_models'], 1):
                print(f"{i}. {question['question']}")
                
                # Show variants if they exist
                if "similar_variants" in question and question["similar_variants"]:
                    print("   Variants:")
                    for j, variant in enumerate(question["similar_variants"], 1):
                        print(f"   {j}. {variant}")
        
        # Show questions agreed by majority of models
        if consensus['majority']:
            print("\nQUESTIONS IDENTIFIED BY MULTIPLE MODELS:")
            print("-"*50)
            for i, question in enumerate(consensus['majority'], 1):
                models = ", ".join(question["identified_by"])
                print(f"{i}. {question['question']} (identified by: {models})")
                
                # Show variants if they exist
                if "similar_variants" in question and question["similar_variants"]:
                    print("   Variants:")
                    for j, variant in enumerate(question["similar_variants"], 1):
                        print(f"   {j}. {variant}")
        
        # Show unique questions by model
        print("\nUNIQUE QUESTIONS BY MODEL:")
        print("-"*50)
        for model_name in self.results["models"].keys():
            unique_questions = consensus.get(f"unique_to_{model_name}", [])
            if unique_questions:
                print(f"\n{model_name.upper()} ({len(unique_questions)} unique):")
                for i, question in enumerate(unique_questions, 1):
                    confidence = question["confidence_scores"].get(model_name, "N/A")
                    confidence_str = f" (confidence: {confidence:.2f})" if isinstance(confidence, (int, float)) else ""
                    print(f"{i}. {question['question']}{confidence_str}")
        
        print("\n" + "="*10)
        print(f"Full results saved to: {self.output_file}")
        print("="*10 + "\n")
    
    def _extract_questions_text(self, model_data: Dict[str, Any]) -> List[str]:
        """Extract just the question text from a model's results."""
        return [q["question"] for q in model_data["questions"]]
    
    def _normalize_question(self, text: str) -> str:
        """
        Normalize a question string for more robust comparison.
        
        This function focuses on removing formatting differences:
        1. Converts to lowercase
        2. Removes extra whitespace
        3. Removes punctuation except question marks
        4. Removes common prefixes
        
        Args:
            text: The question text to normalize
            
        Returns:
            Normalized question text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove beginning expressions like "So," or "And," or "Last two—"
        text = re.sub(r'^(so,?\s+|and,?\s+|last\s+\w+[—-]?\s+|big\s+picture[—-]?\s+|that\'s\s+\w+[—-]?\s+)', '', text)
        
        # Remove punctuation except question marks
        text = re.sub(r'[^\w\s\?]', '', text)
        
        return text

    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate similarity between two strings using Levenshtein distance.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simple Levenshtein distance implementation
        if len(str1) == 0 or len(str2) == 0:
            return 0.0
        
        if len(str1) > len(str2):
            str1, str2 = str2, str1
        
        distances = range(len(str1) + 1)
        for i2, c2 in enumerate(str2):
            distances_ = [i2+1]
            for i1, c1 in enumerate(str1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        
        max_len = max(len(str1), len(str2))
        similarity = 1 - (distances[-1] / max_len if max_len > 0 else 0)
        return similarity

    def _are_questions_similar(self, q1: str, q2: str, threshold: float = 0.9) -> bool:
        """
        Determine if two questions are similar enough to be considered the same.
        
        Args:
            q1: First question text
            q2: Second question text
            threshold: Similarity threshold (0-1)
            
        Returns:
            True if questions are similar enough, False otherwise
        """
        # Normalize both questions
        norm_q1 = self._normalize_question(q1)
        norm_q2 = self._normalize_question(q2)
        
        # If normalized questions are exactly the same, return True
        if norm_q1 == norm_q2:
            return True
        
        # Check word overlap
        words1 = set(norm_q1.split())
        words2 = set(norm_q2.split())
        
        if not words1 or not words2:
            return False
        
        # Calculate overlap ratio
        overlap = len(words1.intersection(words2))
        total = max(len(words1), len(words2))
        overlap_ratio = overlap / total if total > 0 else 0
        
        # If very high overlap, return True 
        if overlap_ratio >= 0.8:
            return True
        
        # For cases with lower overlap, use Levenshtein as a backup
        similarity = self._calculate_string_similarity(norm_q1, norm_q2)
        
        return similarity >= threshold

    def _analyze_consensus(self) -> None:
        """Analyze which questions were identified by multiple models."""
        model_names = list(self.results["models"].keys())
        similar_questions_groups = []
        
        # Extract all questions from all models
        all_model_questions = []
        for model_name in model_names:
            model_data = self.results["models"][model_name]
            for question in model_data["questions"]:
                all_model_questions.append({
                    "text": question["question"],
                    "model": model_name,
                    "confidence": question.get("confidence", None),
                    "context": question.get("context", "")
                })
        
        # Group similar questions
        while all_model_questions:
            current = all_model_questions.pop(0)
            group = [current]
            
            i = 0
            while i < len(all_model_questions):
                if self._are_questions_similar(current["text"], all_model_questions[i]["text"]):
                    group.append(all_model_questions.pop(i))
                else:
                    i += 1
                
            similar_questions_groups.append(group)
        
        # Convert groups to consensus format
        all_questions = {}
        for group in similar_questions_groups:
            # Use the question from the first model as the canonical version
            canonical_question = group[0]["text"]
            identified_by = []
            confidence_scores = {}
            context = group[0]["context"]
            
            for item in group:
                model = item["model"]
                identified_by.append(model)
                confidence_scores[model] = item["confidence"]
                # Use the longest context if available
                if len(item["context"]) > len(context):
                    context = item["context"]
            
            all_questions[canonical_question] = {
                "question": canonical_question,
                "identified_by": identified_by,
                "confidence_scores": confidence_scores,
                "context": context,
                "similar_variants": [item["text"] for item in group if item["text"] != canonical_question]
            }
        
        # Group questions by number of models that identified them
        consensus_groups = {"all_models": [], "majority": [], "unique": []}
        
        for q_text, q_data in all_questions.items():
            if len(q_data["identified_by"]) == len(model_names):
                consensus_groups["all_models"].append(q_data)
            elif len(q_data["identified_by"]) > 1:
                consensus_groups["majority"].append(q_data)
            else:
                consensus_groups["unique"].append(q_data)
        
        # Add model-specific unique questions
        for model_name in model_names:
            model_unique = [q for q in consensus_groups["unique"] if q["identified_by"] == [model_name]]
            consensus_groups[f"unique_to_{model_name}"] = model_unique
        
        self.results["consensus_analysis"] = consensus_groups
    
    def _calculate_performance_metrics(self) -> None:
        """Calculate performance metrics for each model."""
        metrics = {}
        
        for model_name, model_data in self.results["models"].items():
            metrics[model_name] = {
                "processing_time": model_data["metadata"].get("processing_time", None),
                "questions_count": len(model_data["questions"]),
                "avg_confidence": None
            }
            
            # Calculate average confidence if available
            confidences = [q.get("confidence", None) for q in model_data["questions"]]
            valid_confidences = [c for c in confidences if c is not None]
            if valid_confidences:
                metrics[model_name]["avg_confidence"] = sum(valid_confidences) / len(valid_confidences)
        
        self.results["performance_metrics"] = metrics
    
    def _calculate_semantic_similarity(self) -> None:
        """Calculate semantic similarity between questions from different models."""
        model_names = list(self.results["models"].keys())
        similarity_matrix = {}
        
        # Extract questions from each model
        model_questions = {}
        for model_name in model_names:
            model_questions[model_name] = self._extract_questions_text(self.results["models"][model_name])
        
        # Skip if any model has no questions
        if any(len(qs) == 0 for qs in model_questions.values()):
            self.results["semantic_similarity"] = {"error": "One or more models returned no questions"}
            return
        
        # Calculate similarity between models
        for i, model1 in enumerate(model_names):
            similarity_matrix[model1] = {}
            for j, model2 in enumerate(model_names):
                if i == j:  # Same model
                    similarity_matrix[model1][model2] = 1.0
                    continue
                
                # Use TF-IDF and cosine similarity
                tfidf = TfidfVectorizer()
                
                # Combine all questions from both models
                all_questions = model_questions[model1] + model_questions[model2]
                if not all_questions:
                    similarity_matrix[model1][model2] = None
                    continue
                
                try:
                    tfidf_matrix = tfidf.fit_transform(all_questions)
                    
                    # Calculate average similarity between questions from model1 and model2
                    m1_count = len(model_questions[model1])
                    m2_count = len(model_questions[model2])
                    sim_scores = []
                    
                    # For each question from model1, find the best match in model2
                    for k in range(m1_count):
                        best_sim = 0
                        for l in range(m1_count, m1_count + m2_count):
                            # Get documents in the TF-IDF matrix
                            vec1 = tfidf_matrix[k].toarray().flatten()
                            vec2 = tfidf_matrix[l].toarray().flatten()
                            
                            # Calculate cosine similarity
                            sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                            best_sim = max(best_sim, sim)
                        sim_scores.append(best_sim)
                    
                    # Average similarity
                    avg_sim = sum(sim_scores) / len(sim_scores) if sim_scores else 0
                    similarity_matrix[model1][model2] = float(avg_sim)
                except Exception as e:
                    similarity_matrix[model1][model2] = None
                    print(f"⚠️ Error calculating similarity between {model1} and {model2}: {str(e)}")
        
        self.results["semantic_similarity"] = similarity_matrix
    
    def _save_results(self) -> None:
        """Save analysis results to the output file."""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)


def main():
    """Main entry point of the script."""
    parser = argparse.ArgumentParser(description="Analyze open-ended questions using multiple LLM APIs")
    parser.add_argument("input_file", help="Path to the text file to analyze")
    parser.add_argument("-o", "--output", help="Path to save the analysis results (JSON)")
    parser.add_argument("--no-visualize", action="store_true", help="Skip visualization step")
    parser.add_argument("--no-browser", action="store_true", help="Don't open HTML report in browser")
    args = parser.parse_args()
    
    analyzer = OpenEndedQuestionAnalyzer(args.input_file, args.output)
    analyzer.run_analysis()
    
    # Run visualization automatically if not disabled
    if not args.no_visualize:
        output_file = analyzer.output_file
        if os.path.exists(output_file):
            print("\nRunning visualization tool...")
            try:
                # Create visualizations directory if it doesn't exist
                vis_dir = "visualizations"
                os.makedirs(vis_dir, exist_ok=True)
                
                # Run visualization
                subprocess.run(["python", "visualize_results.py", str(output_file)], check=True)
                print("Visualization complete. Check the 'visualizations' directory.")
                
                # Open HTML report in browser
                html_report_path = os.path.join(vis_dir, "analysis_report.html")
                if os.path.exists(html_report_path) and not args.no_browser:
                    print("Opening HTML report in your default browser...")
                    html_report_url = "file://" + os.path.abspath(html_report_path)
                    webbrowser.open(html_report_url)
            except subprocess.CalledProcessError as e:
                print(f"Error running visualization: {e}")
            except FileNotFoundError:
                print("Error: visualize_results.py not found in the current directory.")
        else:
            print(f"Error: Result file {output_file} not found. Skipping visualization.")


if __name__ == "__main__":
    main() 