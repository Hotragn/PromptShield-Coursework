# Fine-Tuning Code — Teammate Instructions

## What this does
Uploads our 20-example training file to OpenAI, fine-tunes gpt-4o-mini,
then runs the fine-tuned model on the same test inputs as the baseline
and compares with evaluation metrics.

## Cost
~$0.25 for training. Plus a few cents for the comparison API calls.

## Where to add
Insert these cells in the main notebook AFTER the Gap 2 cells 
(Baseline vs RAG comparison) and BEFORE the existing Step 4 cells (Cell 66).

## Two-phase process — READ THIS

### Phase 1: Start training (Cells A-C)
1. Run Cell A → uploads quiz_finetune_train.jsonl to OpenAI
2. Run Cell B → creates the fine-tuning job (saves the job ID)
3. Run Cell C → checks training status

**STOP HERE.** Training takes 10-20 minutes. Keep re-running Cell C 
every few minutes until it prints "TRAINING COMPLETE!" and shows 
the fine-tuned model name.

### Phase 2: Run comparison (Cells D-G)  
4. Run Cell D → defines the fine-tuned quiz generator function
5. Run Cell E → generates quizzes with both baseline and fine-tuned model
6. Run Cell F → shows per-example and average score comparison
7. Run Cell G → shows the full three-way comparison (Baseline vs RAG vs Fine-Tuned)

## Dependencies
- All earlier cells must have been run (client, HARDENED_SYSTEM_PROMPT, 
  generate_quiz_baseline, evaluate_quiz_output, parse_eval_scores, 
  test set from stratified split, metrics list)
- Gap 2 must be run first (for rag_all_scores and parse_eval_scores)
- quiz_finetune_train.jsonl must be in the working directory

## If you lose the job ID
If your notebook kernel restarts between Phase 1 and Phase 2:
1. Go to https://platform.openai.com/finetune to find your job
2. Paste the job ID into Cell C where it says FINE_TUNE_JOB_ID = "..."
3. Paste the model name into Cell D where it says FINE_TUNED_MODEL = "..."

## What to screenshot for documentation
- The average score table from Cell F (Baseline vs Fine-Tuned)
- The three-way comparison from Cell G (Baseline vs RAG vs Fine-Tuned)
