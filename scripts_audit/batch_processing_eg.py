#%%
import pandas as pd
import json
from cxr_audit.lib_audit_cxr_v2 import CXRClassifier
from cxr_audit.grade_batch_async import BatchCXRProcessor

with open("padchest_op.json", "r") as f:
    padchest = json.load(f)
    
with open("padchest_tubes_lines.json", "r") as f:
    tubes_lines = json.load(f)

with open("diagnoses.json", "r") as f:
    diagnoses = json.load(f)

# Load data
df = pd.read_csv("../data_audit/op_only.csv")

# Initialize batch processor
processor = BatchCXRProcessor(
    findings_dict=padchest,
    tubes_lines_dict=tubes_lines,
    diagnoses_dict=diagnoses,
    model_name="qwen3:32b-q4_K_M",  # Can change, I used qwen3:32b-q4_K_M
    base_url="http://127.0.0.1:11434/v1",   # Adjust to your Ollama server URL, keep for localhost
    api_key="dummy",
    max_workers=4,  # Adjust based on your system and API limits
    rate_limit_delay=0  # Adjust based on your API rate limits
)

# Process the full pipeline
result_df = processor.process_full_pipeline(df, report_column='TEXT_REPORT')

# Save final results
result_df.to_csv("final_processed_results.csv", index=False)
print("Processing complete! Results saved to final_processed_results.csv")
# %%
