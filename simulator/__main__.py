import argparse
import asyncio
import logging
import uuid

from simulator.config import SimConfig, load_prompt_config, load_sim_config
from simulator.engine import create_engine, get_tokenizer
from simulator.engine_poller import SchedulerStatLogger
from simulator.runner import execute_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("simulator")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VLM GPU profiling simulator for Verkos detection pipeline")
    p.add_argument("--config", required=True, help="Path to sim_config.yaml")
    p.add_argument("--run-id", default=None, help="Override run_id (default: auto UUID)")
    p.add_argument("--output-dir", default=None, help="Override output_dir from config")
    return p.parse_args()


async def run(config: SimConfig, run_id: str) -> None:
    log.info("run_id=%s  model=%s  feeds=%d", run_id, config.model_path, config.num_feeds)

    prompt = load_prompt_config(config.prompts_file)
    log.info("prompts loaded from %s", config.prompts_file)

    log.info("loading engine …")
    stat_logger = SchedulerStatLogger(run_id=run_id)
    engine = create_engine(config, stat_logger=stat_logger)
    tokenizer = await get_tokenizer(engine)
    log.info("engine ready")

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

    run_id = config.run_id or uuid.uuid4().hex[:12]

    asyncio.run(run(config, run_id))


if __name__ == "__main__":
    main()
