#!/usr/bin/env python3
"""
Visualization tool for Open-Ended Question Analyzer results.

This script takes the JSON output from the open_ended_question_analyzer.py
and creates visualizations of the results.
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def load_results(json_file):
    """Load the analysis results from a JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def visualize_question_counts(results, output_dir):
    """Visualize the number of questions identified by each model."""
    models = list(results["models"].keys())
    counts = [len(results["models"][model]["questions"]) for model in models]
    
    plt.figure(figsize=(10, 6))
    plt.bar(models, counts, color=sns.color_palette("muted"))
    plt.title("Number of Open-Ended Questions Identified per Model")
    plt.ylabel("Number of Questions")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_path = Path(output_dir) / "question_counts.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Created visualization: {output_path}")


def visualize_consensus(results, output_dir):
    """Visualize the consensus between models."""
    consensus = results["consensus_analysis"]
    categories = ["all_models", "majority", "unique"]
    counts = [len(consensus[cat]) for cat in categories]
    
    # Add model-specific unique questions
    models = list(results["models"].keys())
    for model in models:
        categories.append(f"unique_to_{model}")
        counts.append(len(consensus.get(f"unique_to_{model}", [])))
    
    plt.figure(figsize=(12, 6))
    plt.bar(categories, counts, color=sns.color_palette("muted", len(categories)))
    plt.title("Consensus Analysis of Open-Ended Questions")
    plt.ylabel("Number of Questions")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_path = Path(output_dir) / "consensus_analysis.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Created visualization: {output_path}")


def visualize_confidence_scores(results, output_dir):
    """Visualize the average confidence scores (if available)."""
    models = list(results["models"].keys())
    scores = []
    
    for model in models:
        score = results["performance_metrics"][model]["avg_confidence"]
        scores.append(score if score is not None else 0)
    
    plt.figure(figsize=(10, 6))
    plt.bar(models, scores, color=sns.color_palette("muted"))
    plt.title("Average Confidence Score per Model")
    plt.ylabel("Confidence Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_path = Path(output_dir) / "confidence_scores.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Created visualization: {output_path}")


def generate_html_report(results, output_dir):
    """Generate an HTML report summarizing the analysis results."""
    file_analyzed = results["file"]
    models = list(results["models"].keys())
    consensus = results["consensus_analysis"]
    
    # Create the HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Open-Ended Question Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ padding: 8px; text-align: left; border: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            .highlight {{ background-color: #ffeeba; }}
            .section {{ margin-bottom: 30px; }}
            .image-container {{ margin: 20px 0; text-align: center; }}
            .image-container img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Open-Ended Question Analysis Report</h1>
            <p>File analyzed: <strong>{file_analyzed}</strong></p>
            
            <div class="section">
                <h2>Summary</h2>
                <table>
                    <tr>
                        <th>Model</th>
                        <th>Questions Found</th>
                        <th>Avg. Confidence</th>
                    </tr>
    """
    
    # Add model summary rows
    for model in models:
        questions_count = len(results["models"][model]["questions"])
        avg_confidence = results["performance_metrics"][model]["avg_confidence"]
        avg_confidence_str = f"{avg_confidence:.2f}" if avg_confidence is not None else "N/A"
        
        html_content += f"""
                    <tr>
                        <td>{model}</td>
                        <td>{questions_count}</td>
                        <td>{avg_confidence_str}</td>
                    </tr>
        """
    
    html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                
                <div class="image-container">
                    <h3>Questions Identified per Model</h3>
                    <img src="question_counts.png" alt="Questions per Model Chart">
                </div>
                
                <div class="image-container">
                    <h3>Consensus Analysis</h3>
                    <img src="consensus_analysis.png" alt="Consensus Analysis Chart">
                </div>
                
                <div class="image-container">
                    <h3>Average Confidence Scores</h3>
                    <img src="confidence_scores.png" alt="Confidence Scores Chart">
                </div>
            </div>
            
            <div class="section">
                <h2>Consensus Details</h2>
                
                <h3>Questions Identified by All Models ({len(consensus['all_models'])})</h3>
                <table>
                    <tr>
                        <th>Question</th>
                        <th>Context</th>
                    </tr>
    """
    
    # Add questions identified by all models
    for question in consensus["all_models"]:
        html_content += f"""
                    <tr>
                        <td>{question['question']}</td>
                        <td>{question.get('context', '')}</td>
                    </tr>
        """
    
    html_content += """
                </table>
                
                <h3>Questions Identified by Multiple Models ({len(consensus['majority'])})</h3>
                <table>
                    <tr>
                        <th>Question</th>
                        <th>Identified By</th>
                        <th>Context</th>
                    </tr>
    """
    
    # Add questions identified by majority of models
    for question in consensus["majority"]:
        identified_by = ", ".join(question["identified_by"])
        html_content += f"""
                    <tr>
                        <td>{question['question']}</td>
                        <td>{identified_by}</td>
                        <td>{question.get('context', '')}</td>
                    </tr>
        """
    
    html_content += """
                </table>
                
                <h3>Unique Questions (Only Identified by One Model)</h3>
    """
    
    # Add unique questions by model
    for model in models:
        model_unique = consensus.get(f"unique_to_{model}", [])
        if not model_unique:
            continue
            
        html_content += f"""
                <h4>Unique to {model} ({len(model_unique)})</h4>
                <table>
                    <tr>
                        <th>Question</th>
                        <th>Confidence</th>
                        <th>Context</th>
                    </tr>
        """
        
        for question in model_unique:
            confidence = question["confidence_scores"].get(model, "N/A")
            confidence_str = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "N/A"
            html_content += f"""
                    <tr>
                        <td>{question['question']}</td>
                        <td>{confidence_str}</td>
                        <td>{question.get('context', '')}</td>
                    </tr>
            """
        
        html_content += """
                </table>
        """
    
    # Add variants section if available
    has_variants = False
    for cat in ["all_models", "majority", "unique"]:
        for question in consensus[cat]:
            if "similar_variants" in question and question["similar_variants"]:
                has_variants = True
                break
        if has_variants:
            break
            
    if has_variants:
        html_content += """
            <div class="section">
                <h2>Variant Questions</h2>
                <p>These are different phrasings of the same question identified by different models.</p>
                <table>
                    <tr>
                        <th>Canonical Question</th>
                        <th>Variants</th>
                        <th>Models</th>
                    </tr>
        """
        
        for cat in ["all_models", "majority", "unique"]:
            for question in consensus[cat]:
                if "similar_variants" in question and question["similar_variants"]:
                    variants = "<br>".join(question["similar_variants"])
                    models = ", ".join(question["identified_by"])
                    html_content += f"""
                        <tr>
                            <td>{question['question']}</td>
                            <td>{variants}</td>
                            <td>{models}</td>
                        </tr>
                    """
                    
        html_content += """
                </table>
            </div>
        """
    
    html_content += """
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write the HTML report to file
    output_path = Path(output_dir) / "analysis_report.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Created HTML report: {output_path}")


def main():
    """Main entry point of the script."""
    parser = argparse.ArgumentParser(description="Visualize Open-Ended Question Analyzer results")
    parser.add_argument("input_file", help="Path to the JSON results file")
    parser.add_argument("-o", "--output_dir", default="visualizations", help="Directory to save visualizations")
    args = parser.parse_args()
    
    # Load the results
    results = load_results(args.input_file)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Generating visualizations in: {output_dir}")
    
    # Create visualizations
    visualize_question_counts(results, output_dir)
    visualize_consensus(results, output_dir)
    visualize_confidence_scores(results, output_dir)
    
    # Generate HTML report
    generate_html_report(results, output_dir)
    
    print("All visualizations completed!")


if __name__ == "__main__":
    main() 