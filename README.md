# Creation Of a Large Language Model-Based Review Tool for Follow-Up Of Chest X-Rays

## Project Summary  
This project implements a system for automating the grading of chest X-ray reports using Large Language Models (LLMs). The goal is to classify reports into clinically relevant categories based on the urgency of follow-up required, ranging from normal (grade 1) to critical (grade 5). The system aims to streamline radiology workflow by automating report categorization while preserving the advantages of human oversight.

Two complementary approaches were developed and evaluated:

1. **Semi-Algorithmic Approach**: Extracts structured findings from reports using an LLM, then applies rule-based grading based on finding types, temporal changes (new/stable/worsening), uncertainty levels, and medical device positioning.  

2. **All-LLM Approach**: Leverages an LLM to directly grade reports based on provided guidelines without the intermediate structured extraction step.

3. **Hybrid Method**: Combines semi-algorithmic results with LLM judgment ("second-think") to come to a reasoned grade

4. **Judge Method**: Evaluates the semialgo (1) and LLM (2) approaches to determine which is more appropriate as an ensemble system.

## Features

- **Multiple Grading Methods**: Semi-algorithmic, pure LLM, hybrid, and judge-based approaches
- **Concurrent Processing**: High-performance batch processing with configurable worker threads
- **Flexible Model Support**: Ollama used for local LLM server through its OpenAI-compatible API, ensuring privacy (though the data is still anonymised)

## Repository Contents  
- scripts_audit/audit_cxr_v2.py: Python script implementing the grading approaches  
- scripts_audit/results_analysis.ipynb: Jupyter notebook containing analysis of the grading results, including statistical evaluations and visualizations  

## Implementation Details
Grading System (R)  
- R1: Normal without any findings  
- R2: Normal variant or minor pathology, does not require follow-up  
- R3: Abnormal, non-urgent follow-up required 
- R4: Abnormal, potentially important finding  
- R5: Critical, urgent follow-up required  
  
Data and Models  
- Dataset: 1114 anonymised primary care CXR reports
- Ground Truth: Expert radiologist annotation
- Models: Qwen3 32B, quantized to Q4_K_M, temperature=0.2, running on Ollama
- Evaluation Metrics: Exact agreement, Cohen's Kappa, sensitivity, specificity, F1 score, ROC-AUC

## Results
For the purposes of this project, the grades were further subdivided into 0: Normal (1-2), 1: Actionable (3, 4), 2: Critical (5).
All four approaches yielded good results, with a Cohen Kappa of up to 0.928 for the LLM approach and 0.846 for the Hybrid approach. 

If we further group the metrics into normal (0) and abnormal (everything else), the Cohen Kappa remains the same. The ROC-AUC of the LLM approach was 0.966 and the ROC-AUC of the Hybrid approach was 0.933.

McNemar's test revealed a significant difference between the accuracy of the Hybrid and the Judge approach (p=0.004).

## Technical Details

### Installation

#### Prerequisites
- Python 3.8+
- Access to OpenAI API or local Ollama installation

#### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/covie154/CXR-Audit
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare configuration files:
   - `padchest_op.json`: Medical findings dictionary
   - `padchest_tubes_lines.json`: Tubes and lines dictionary  
   - `diagnoses.json`: Diagnoses dictionary

### Usage

#### Basic Example
See batch_processing_eg.py for a basic example

```python
import pandas as pd
import json
from cxr_audit.grade_batch_async import BatchCXRProcessor

# Load configuration dictionaries
with open("padchest_op.json", "r") as f:
    padchest = json.load(f)
    
with open("padchest_tubes_lines.json", "r") as f:
    tubes_lines = json.load(f)

with open("diagnoses.json", "r") as f:
    diagnoses = json.load(f)

# Load your data
df = pd.read_csv("your_data.csv")

# Initialize batch processor
processor = BatchCXRProcessor(
    findings_dict=padchest,
    tubes_lines_dict=tubes_lines,
    diagnoses_dict=diagnoses,
    model_name="gpt-4o-mini",  # or "qwen3:32b-q4_K_M" for Ollama
    base_url="https://api.openai.com/v1",  # or "http://localhost:11434/v1" for Ollama
    api_key="your-api-key",  # or "dummy" for Ollama
    max_workers=4,
    rate_limit_delay=0.1
)

# Process the full pipeline
result_df = processor.process_full_pipeline(df, report_column='REPORT_TEXT')

# Save results
result_df.to_csv("graded_reports.csv", index=False)
```

#### Individual Method Usage

```python
# Process only specific methods
df_with_semialgo = processor.process_semialgo_batch(df, report_column='REPORT_TEXT')
df_with_llm = processor.process_llm_batch(df, report_column='REPORT_TEXT')
df_with_hybrid = processor.process_hybrid_batch(df, report_column='REPORT_TEXT')
df_with_judge = processor.process_judge_batch(df, report_column='REPORT_TEXT')
```

#### Single Report Processing

```python
# Initialize single classifier
classifier = CXRClassifier(
    findings=padchest,
    tubes_lines=tubes_lines,
    diagnoses=diagnoses,
    model_name="gpt-4o-mini"
)

# Grade single report
report_text = "Your chest X-ray report text here"
semialgo_result = classifier.gradeReportSemialgo(report_text)
llm_result = classifier.gradeReportLLM(report_text)
hybrid_result = classifier.gradeReportHybrid(report_text, semialgo_grade=3)
```

### Configuration

#### Model Configuration
- **OpenAI Models**: Any compatible model (e.g. `o4-mini`)
- **Ollama Models**: Any compatible model (e.g., `qwen3:32b-q4_K_M`, `llama3.1:8b`)

#### Performance Tuning
- `max_workers`: Number of concurrent processing threads (default: 5)
- `rate_limit_delay`: Delay between API calls in seconds (default: 0.1)

#### Data Format
Your input DataFrame should contain:
- A text column with the chest X-ray reports
- Optional ground truth columns for evaluation

### File Structure

```
scripts_audit/
├──cxr_audit/
├──├── lib_audit_cxr_v2.py      # Core CXRClassifier implementation
├──├── grade_batch_async.py     # Batch processing with concurrency
├──├── helpers.py               # Utility functions
├──├── prompts.py              # LLM prompts and templates
├──batch_processing_eg.py       # Example usage script
├──requirements.txt            # Python dependencies
```

### Library Reference

#### BatchCXRProcessor

Main class for concurrent batch processing of CXR reports.

##### Methods
- `process_full_pipeline(df, report_column)`: Run complete grading pipeline
- `process_semialgo_batch(df, report_column)`: Semi-algorithmic grading only
- `process_llm_batch(df, report_column)`: Pure LLM grading only
- `process_hybrid_batch(df, report_column)`: Hybrid grading only
- `process_judge_batch(df, report_column)`: Judge-based comparison only

#### CXRClassifier

Core classifier for individual report processing.

##### Methods
- `gradeReportSemialgo(report_text)`: Semi-algorithmic approach
- `gradeReportLLM(report_text)`: Pure LLM approach
- `gradeReportHybrid(report_text, semialgo_grade)`: Hybrid approach
- `gradeReportJudge(report_text, manual_grade, algo_grade, llm_grade)`: Judge approach

