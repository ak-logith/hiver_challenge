import time
import sys
import io

# Ensure UTF-8 output encoding on Windows consoles
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from fetch_dataset import fetch_and_sample_dataset
from generator import run_generation
from evaluator import run_evaluation

def main():
    start_time = time.time()
    print("=" * 65)
    print(">> AI EMAIL REPLY GENERATOR & EVALUATOR PIPELINE")
    print("=" * 65)

    try:
        # Step 1: Dataset loading / sampling
        print("\n[STEP 1/3] Fetching & sampling customer support dataset...")
        t0 = time.time()
        emails = fetch_and_sample_dataset()
        print(f"[OK] Step 1 completed in {time.time() - t0:.2f}s ({len(emails)} emails loaded)\n")

        # Step 2: Generation
        print("[STEP 2/3] Generating AI replies for customer emails...")
        t1 = time.time()
        replies = run_generation()
        print(f"[OK] Step 2 completed in {time.time() - t1:.2f}s ({len(replies)} replies generated)\n")

        # Step 3: Evaluation
        print("[STEP 3/3] Evaluating replies (LLM-as-a-judge rubric + lexical overlap)...")
        t2 = time.time()
        scores = run_evaluation()
        print(f"[OK] Step 3 completed in {time.time() - t2:.2f}s\n")

        total_elapsed = time.time() - start_time
        summary = scores["summary"]
        print("=" * 65)
        print(">> PIPELINE COMPLETED SUCCESSFULLY")
        print(f"Total time elapsed   : {total_elapsed:.2f}s")
        print(f"Total emails scored  : {summary['total_evaluated']}")
        print(f"Mean composite score : {summary['composite']['mean']} / 100")
        print(f"Min / Max score      : {summary['composite']['min']} / {summary['composite']['max']}")
        print("To explore the interactive dashboard, run:")
        print("  streamlit run app.py")
        print("=" * 65)

    except Exception as exc:
        print(f"\n[ERROR] Pipeline failed with error: {exc}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
