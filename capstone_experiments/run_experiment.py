import os
import json
import logging
import argparse
from src.config import ExperimentConfig
from src.model import GeminiClient
from src.evaluation import EvaluationLoader
from src.metrics import ExperimentMetricsCollector
from src.storage import ExperimentResultStorage
from src.strategies.base_strategy import (
    PromptOnlyStrategy,
    RAGStrategy,
    MemoryAugmentedPromptingStrategy,
    LifecycleGuidedContextStrategy
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("experiment_runner")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Context Construction Experiment")
    parser.add_argument(
        "--strategy",
        required=True,
        choices=["prompt-only", "rag", "memory", "lifecycle"],
        help="Experimental condition strategy to run."
    )
    parser.add_argument(
        "--config",
        default="config/default_config.json",
        help="Path to experiment config JSON."
    )
    parser.add_argument(
        "--tasks",
        default="data/evaluation_tasks.json",
        help="Path to evaluation tasks JSON."
    )
    return parser.parse_args()

def get_strategy_instance(strategy_name: str):
    mapping = {
        "prompt-only": PromptOnlyStrategy,
        "rag": RAGStrategy,
        "memory": MemoryAugmentedPromptingStrategy,
        "lifecycle": LifecycleGuidedContextStrategy
    }
    return mapping[strategy_name.lower()]()

def print_final_status_table():
    """Prints a summary table showing the execution and implementation status of all four strategies."""
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    
    # Check which result archives exist
    has_prompt = any(f.startswith("run_prompt-only_") for f in os.listdir(results_dir)) if os.path.exists(results_dir) else False
    has_rag = any(f.startswith("run_rag_") for f in os.listdir(results_dir)) if os.path.exists(results_dir) else False
    has_mem = any(f.startswith("run_memory-augmented_") for f in os.listdir(results_dir)) if os.path.exists(results_dir) else False
    has_life = any(f.startswith("run_lifecycle-guided_") for f in os.listdir(results_dir)) if os.path.exists(results_dir) else False

    def check_emoji(condition: bool) -> str:
        return "✅" if condition else "❌"

    print("\n" + "="*80)
    print("FINAL CAPSTONE STRATEGY STATUS SUMMARY")
    print("="*80)
    print("| Context Construction Strategy | Implemented | Executed | Available through CLI | Results Archived |")
    print("|" + "-"*31 + "|" + "-"*13 + "|" + "-"*10 + "|" + "-"*23 + "|" + "-"*18 + "|")
    print(f"| Prompt-only interactions      |     ✅      |    {check_emoji(has_prompt)}     |          ✅           |        {check_emoji(has_prompt)}        |")
    print(f"| Retrieval-Augmented Gen (RAG) |     ✅      |    {check_emoji(has_rag)}     |          **✅**           |        {check_emoji(has_rag)}        |")
    print(f"| Memory-Augmented Prompting    |     ✅      |    {check_emoji(has_mem)}     |          ✅           |        {check_emoji(has_mem)}        |")
    print(f"| Lifecycle-Guided Strategy     |     ✅      |    {check_emoji(has_life)}     |          ✅           |        {check_emoji(has_life)}        |")
    print("="*80 + "\n")

def main():
    args = parse_arguments()
    logger.info(f"Setting up experiment for strategy: {args.strategy}")

    # 1. Load configurations
    config = ExperimentConfig(args.config)
    
    # 2. Initialize Gemini Client wrapper
    # Load API key from environment variable when real API mode is enabled
    api_key = os.environ.get("GEMINI_API_KEY", config.api_key)
    client = GeminiClient(
        model_name=config.gemini_model_name,
        api_key=api_key,
        temperature=config.temperature,
        max_tokens=config.max_output_tokens,
        mock_mode=config.mock_mode
    )

    # 3. Load evaluation tasks
    eval_loader = EvaluationLoader(args.tasks)
    tasks = eval_loader.load_tasks()

    # 4. Instantiate strategy
    strategy = get_strategy_instance(args.strategy)
    logger.info(f"Instantiated strategy: {strategy.name}")

    # 5. Initialize helper modules
    metrics_collector = ExperimentMetricsCollector()
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    storage = ExperimentResultStorage(results_dir)

    # Reset session memory at the beginning of a fresh run
    memory_path = os.path.join(results_dir, "session_memory.json")
    if os.path.exists(memory_path):
        try:
            os.remove(memory_path)
            logger.info("Cleared prior session memory for fresh run.")
        except Exception as e:
            logger.warning(f"Could not clear session memory file: {e}")

    # 6. Run loop
    detailed_steps = []
    for task in tasks:
        task_id = task["task_id"]
        logger.info(f"Executing task {task_id} on version tag {task['target_version']}...")

        # Step 1: Construct context block using the selected strategy
        context = strategy.construct_context(task, config.dataset_dir)

        # Step 2: Query Gemini API
        prompt_query = task["description"]
        result = client.call_gemini(prompt=prompt_query, context=context)

        # Step 3: Record performance metrics
        metrics_collector.record_step(
            task_id=task_id,
            strategy=strategy.name,
            latency=result["latency_seconds"],
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            context_len=len(context)
        )

        # Append details for archiving
        detailed_steps.append({
            "task": task,
            "constructed_context": context,
            "response_text": result["response_text"],
            "metrics": {
                "latency_seconds": result["latency_seconds"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"]
            }
        })

        # Save context memory if strategy is Memory-Augmented Prompting
        if args.strategy == "memory":
            memory_data = []
            if os.path.exists(memory_path):
                try:
                    with open(memory_path, "r") as mf:
                        memory_data = json.load(mf)
                except Exception:
                    pass
            memory_data.append({
                "task_id": task_id,
                "description": task["description"],
                "response_summary": result["response_text"][:250].replace("\n", " ") + "..."
            })
            try:
                with open(memory_path, "w") as mf:
                    json.dump(memory_data, mf, indent=2)
            except Exception as e:
                logger.warning(f"Failed to update session memory file: {e}")

    # 7. Compute and print statistics
    summary_stats = metrics_collector.compute_summary_statistics()
    
    # Save the run details
    saved_file = storage.save_run(
        strategy_name=strategy.name,
        config=config.to_dict(),
        steps=detailed_steps,
        summary_stats=summary_stats
    )

    # Output simple results table
    print("\n" + "="*50)
    print(f"EXPERIMENTAL RUN SUMMARY: {strategy.name.upper()}")
    print("="*50)
    print(f"Total tasks processed: {summary_stats.get('total_runs')}")
    print(f"Mean Latency:          {summary_stats.get('latency', {}).get('mean', 0.0):.2f} seconds")
    print(f"Total Input Tokens:    {summary_stats.get('tokens', {}).get('total_input', 0)}")
    print(f"Total Output Tokens:   {summary_stats.get('tokens', {}).get('total_output', 0)}")
    print(f"Mean Context Size:     {summary_stats.get('context_length', {}).get('mean_chars', 0.0):.1f} characters")
    print(f"Result Archive:        {saved_file}")
    print("="*50 + "\n")

    # Output final status matrix
    print_final_status_table()

if __name__ == "__main__":
    main()
