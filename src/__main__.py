import argparse
import asyncio
import logging
import uuid
from datetime import datetime

from src.config import SimConfig, load_prompt_config, load_sim_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("src")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VLM GPU profiling simulator for Verkos detection pipeline")
    p.add_argument("--config", required=True, help="Path to sim_config.yaml")
    p.add_argument("--run-id", default=None, help="Override run_id (default: auto UUID)")
    p.add_argument("--output-dir", default=None, help="Override output_dir from config")
    return p.parse_args()


async def run(config: SimConfig, run_id: str) -> None:
    from src.runner import execute_run

    prompt = load_prompt_config(config.prompts_file)
    log.info("prompts loaded from %s", config.prompts_file)

    engine = None
    tokenizer = None
    stat_logger = None

    if config.inference_mode == "local":
        from src.engine import create_engine, get_tokenizer
        from src.engine_poller import SchedulerStatLogger

        log.info("inference_mode=local  model=%s  feeds=%d", config.model_path, config.num_feeds)
        log.info("loading engine …")
        stat_logger = SchedulerStatLogger(run_id=run_id)
        engine = create_engine(config, stat_logger=stat_logger)
        tokenizer = await get_tokenizer(engine)
        log.info("engine ready")
    else:
        assert config.openrouter is not None
        log.info("inference_mode=openrouter  model=%s  feeds=%d", config.openrouter.model, config.num_feeds)

    arts = await execute_run(config, run_id, engine, tokenizer, prompt, stat_logger)

    for key, path in arts.paths.items():
        log.info("%-8s → %s", key, path)

    print("\n" + arts.paths["summary"].read_text())


def main() -> None:
    args = _parse_args()
    config: SimConfig = load_sim_config(args.config)

    if args.run_id:
        config = config.model_copy(update={"run_id": args.run_id})
    if args.output_dir:
        config = config.model_copy(update={"output_dir": args.output_dir})

    run_id = config.run_id or f"{uuid.uuid4().hex[:8]}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    asyncio.run(run(config, run_id))


if __name__ == "__main__":
    main()
