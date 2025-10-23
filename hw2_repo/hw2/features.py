import typer
from loguru import logger

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    # input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    # -----------------------------------------


if __name__ == "__main__":
    app()
