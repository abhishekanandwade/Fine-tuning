# Path should be inside the code-reviw

# 1 simple RAG
python pipeline/review_pipeline.py --repo "../template" --output "./results/test.json" --mode rag-only --ollama-model qwen2.5-coder:7b

# 2 Agentic RAG
python pipeline/review_pipeline.py --repo "../template" --output "./results/test_agentic.json" --mode rag-only --rag-mode agentic --ollama-model qwen2.5-coder:7b

# Full help
python pipeline/review_pipeline.py --help

# Other modes
--mode hybrid           # RAG + fine-tuned model
--mode fine-tune-only   # Fine-tuned model without RAG

# Other Ollama models you have
--ollama-model deepseek-coder-v2:16b
--ollama-model llama3.2:latest


# Evaluation
# 1. makes the model review code and produce findings using simple RAG;

python pipeline/review_pipeline.py `
    --repo ./benchmarks/seeded_repo `
    --output ./results/seeded_review_simple.json `
    --mode rag-only `
    --ollama-model qwen2.5-coder:7b

# 2. makes the model review code and produce findings using Agentic RAG;

cd code-review

python pipeline/review_pipeline.py `
    --repo ./benchmarks/seeded_repo `
    --output ./results/seeded_review_agentic.json `
    --mode rag-only `
    --rag-mode agentic `
    --ollama-model qwen2.5-coder:7b

# 3. compares those findings to known-correct answers and prints an accuracy grade.

# cd code-review; 

python training/evaluate_seeded.py 
    --review ./results/seeded_review.json 
    --ground-truth ./benchmarks/ground_truth.json 
    --report ./results/seeded_eval_report.json

# from code-review/
python pipeline/review_pipeline.py --repo ./benchmarks/seeded_repo `
    --output ./results/seeded_review_simple.json `
    --mode rag-only --ollama-model qwen2.5-coder:7b

python pipeline/review_pipeline.py --repo ./benchmarks/seeded_repo `
    --output ./results/seeded_review_agentic.json `
    --mode rag-only --rag-mode agentic --ollama-model qwen2.5-coder:7b

# Then grade both against ground truth
python training/evaluate_seeded.py --review ./results/seeded_review_simple.json  --ground-truth ./benchmarks/ground_truth.json --report ./results/eval_simple.json

python training/evaluate_seeded.py --review ./results/seeded_review_agentic.json --ground-truth ./benchmarks/ground_truth.json --report ./results/eval_agentic.json