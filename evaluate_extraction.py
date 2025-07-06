#!/usr/bin/env python3
"""
Comprehensive PDF Extraction Evaluation Script
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from collections import defaultdict

class ExtractionEvaluator:
    def __init__(self):
        self.expected_fields = [
            'fund_manager', 'aum', 'nav', 'inception_date', 'risk', 
            'portfolio', 'scheme_code', 'ratings', 'benchmark', 
            'exit_load', 'min_investment', 'contact', 'tables'
        ]
        
    def evaluate_single_file(self, json_path: str) -> Dict[str, Any]:
        """Evaluate a single extracted JSON file"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            return {
                'file': json_path,
                'error': str(e),
                'completeness': 0,
                'accuracy': 0,
                'structure_quality': 0,
                'overall_score': 0
            }
        
        # Completeness: How many expected fields are present
        present_fields = [field for field in self.expected_fields if field in data and data[field]]
        completeness = len(present_fields) / len(self.expected_fields)
        
        # Structure Quality: How well-structured is the data
        structure_score = 0
        if 'fund_manager' in data and isinstance(data['fund_manager'], list) and len(data['fund_manager']) > 0:
            structure_score += 0.3
        if 'tables' in data and isinstance(data['tables'], list):
            structure_score += 0.2
        if any(isinstance(data.get(field), (list, dict)) for field in ['portfolio', 'ratings']):
            structure_score += 0.2
        if any(isinstance(data.get(field), str) for field in ['aum', 'nav', 'inception_date']):
            structure_score += 0.3
        
        # Accuracy: Check for common data quality issues
        accuracy_score = 1.0
        if 'fund_manager' in data:
            if isinstance(data['fund_manager'], list) and len(data['fund_manager']) > 0:
                # Check if fund manager data looks reasonable
                for manager in data['fund_manager']:
                    if isinstance(manager, dict) and 'name' in manager:
                        if len(manager['name']) < 3:  # Too short
                            accuracy_score -= 0.1
                    elif isinstance(manager, str) and len(manager) < 3:
                        accuracy_score -= 0.1
        
        # Check for empty or null values
        empty_fields = sum(1 for field in self.expected_fields 
                          if field in data and (data[field] is None or data[field] == [] or data[field] == {}))
        if empty_fields > len(self.expected_fields) * 0.7:  # More than 70% empty
            accuracy_score -= 0.3
        
        accuracy_score = max(0, accuracy_score)
        
        # Overall score (weighted average)
        overall_score = (completeness * 0.4 + structure_score * 0.3 + accuracy_score * 0.3)
        
        return {
            'file': json_path,
            'completeness': round(completeness, 3),
            'structure_quality': round(structure_score, 3),
            'accuracy': round(accuracy_score, 3),
            'overall_score': round(overall_score, 3),
            'present_fields': present_fields,
            'missing_fields': [f for f in self.expected_fields if f not in present_fields],
            'field_count': len(present_fields),
            'total_expected': len(self.expected_fields)
        }
    
    def evaluate_all_files(self, json_dir: str) -> Dict[str, Any]:
        """Evaluate all JSON files in a directory"""
        json_dir = Path(json_dir)
        json_files = list(json_dir.glob("*.json"))
        
        if not json_files:
            return {"error": "No JSON files found"}
        
        results = []
        for json_file in json_files:
            result = self.evaluate_single_file(str(json_file))
            results.append(result)
        
        # Aggregate statistics
        avg_completeness = sum(r['completeness'] for r in results if 'completeness' in r) / len(results)
        avg_structure = sum(r['structure_quality'] for r in results if 'structure_quality' in r) / len(results)
        avg_accuracy = sum(r['accuracy'] for r in results if 'accuracy' in r) / len(results)
        avg_overall = sum(r['overall_score'] for r in results if 'overall_score' in r) / len(results)
        
        # Field coverage analysis
        field_coverage = defaultdict(int)
        for result in results:
            if 'present_fields' in result:
                for field in result['present_fields']:
                    field_coverage[field] += 1
        
        return {
            'summary': {
                'total_files': len(results),
                'avg_completeness': round(avg_completeness, 3),
                'avg_structure_quality': round(avg_structure, 3),
                'avg_accuracy': round(avg_accuracy, 3),
                'avg_overall_score': round(avg_overall, 3)
            },
            'field_coverage': dict(field_coverage),
            'detailed_results': results
        }
    
    def generate_report(self, evaluation_results: Dict[str, Any], output_file: str = None):
        """Generate a detailed evaluation report"""
        if 'error' in evaluation_results:
            print(f"❌ Error: {evaluation_results['error']}")
            return
        
        summary = evaluation_results['summary']
        field_coverage = evaluation_results['field_coverage']
        detailed_results = evaluation_results['detailed_results']
        
        report = f"""
# PDF Extraction Evaluation Report

## 📊 Summary Statistics
- **Total Files Processed**: {summary['total_files']}
- **Average Completeness**: {summary['avg_completeness']:.1%}
- **Average Structure Quality**: {summary['avg_structure_quality']:.1%}
- **Average Accuracy**: {summary['avg_accuracy']:.1%}
- **Overall Score**: {summary['avg_overall_score']:.1%}

## 🎯 Field Coverage Analysis
"""
        
        for field in self.expected_fields:
            coverage = field_coverage.get(field, 0)
            percentage = (coverage / summary['total_files']) * 100
            report += f"- **{field.replace('_', ' ').title()}**: {coverage}/{summary['total_files']} ({percentage:.1f}%)\n"
        
        report += "\n## 📋 Detailed Results\n"
        
        for result in detailed_results:
            if 'error' in result:
                report += f"\n### ❌ {Path(result['file']).name}\n"
                report += f"Error: {result['error']}\n"
            else:
                report += f"\n### 📄 {Path(result['file']).name}\n"
                report += f"- **Overall Score**: {result['overall_score']:.1%}\n"
                report += f"- **Completeness**: {result['completeness']:.1%}\n"
                report += f"- **Structure Quality**: {result['structure_quality']:.1%}\n"
                report += f"- **Accuracy**: {result['accuracy']:.1%}\n"
                report += f"- **Fields Found**: {result['field_count']}/{result['total_expected']}\n"
                
                if result['missing_fields']:
                    report += f"- **Missing Fields**: {', '.join(result['missing_fields'])}\n"
        
        # Recommendations
        report += "\n## 💡 Recommendations\n"
        if summary['avg_completeness'] < 0.5:
            report += "- ⚠️ **Low Completeness**: Consider improving extraction logic for missing fields\n"
        if summary['avg_structure_quality'] < 0.5:
            report += "- ⚠️ **Poor Structure**: Focus on better data structuring and parsing\n"
        if summary['avg_accuracy'] < 0.7:
            report += "- ⚠️ **Accuracy Issues**: Review extraction accuracy and data validation\n"
        
        if summary['avg_overall_score'] >= 0.8:
            report += "- ✅ **Excellent Extraction Quality**\n"
        elif summary['avg_overall_score'] >= 0.6:
            report += "- ✅ **Good Extraction Quality**\n"
        else:
            report += "- ❌ **Needs Improvement**\n"
        
        print(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"\n📄 Report saved to: {output_file}")
        
        return report

def main():
    evaluator = ExtractionEvaluator()
    
    # Evaluate all extracted JSON files
    json_dir = "extracted_json"
    
    if not os.path.exists(json_dir):
        print(f"❌ Directory {json_dir} not found!")
        return
    
    print("🔍 Evaluating PDF extraction quality...")
    results = evaluator.evaluate_all_files(json_dir)
    
    # Generate and display report
    evaluator.generate_report(results, "extraction_evaluation_report.md")
    
    # Also save detailed results as JSON
    with open("extraction_evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print("📊 Detailed results saved to: extraction_evaluation_results.json")

if __name__ == "__main__":
    main() 