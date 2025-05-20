import argparse
import os
from typing import List, Optional

from src.analyzer import Analyzer
from src.currency_exchange import Exchanger
from src.data_collector import DataCollector
from src.parser import Settings
from src.predictor import Predictor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process multiple professions for HeadHunter research.")
    parser.add_argument("--config", type=str, default="settings.json", help="Path to config file.")
    parser.add_argument(
        "--professions_file",
        type=str,
        default="professions.txt",
        help="Path to file with professions list (one profession per line).",
    )
    parser.add_argument("--professions", type=str, nargs="+", help="List of professions to process (overrides file).")
    parser.add_argument("--base_dir", type=str, default="data", help="Base directory to store results.")
    return parser.parse_args()


def read_professions_file(file_path: str) -> List[str]:
    """Read professions from a file, one per line."""
    if not os.path.exists(file_path):
        print(f"[WARNING]: File {file_path} not found.")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        # Читаем строки, удаляем пробелы в начале и конце, игнорируем пустые строки
        return [line.strip() for line in f if line.strip()]


class MultiProcessor:
    """Class for processing multiple professions from a list."""

    def __init__(
        self,
        config_path: str = "settings.json",
        professions_list: Optional[List[str]] = None,
        professions_file: str = "professions.txt",
        base_dir: str = "data",
    ):
        """
        Initialize the MultiProcessor.

        Parameters
        ----------
        config_path : str
            Path to the configuration file.
        professions_list : List[str], optional
            List of professions to process. If None, read from professions_file.
        professions_file : str
            Path to file with list of professions.
        base_dir : str
            Base directory to store results.
        """
        self.config_path = config_path
        self.settings = Settings(config_path, no_parse=True)
        self.base_dir = base_dir

        # Use provided list or read from file
        if professions_list:
            self.professions_list = professions_list
        else:
            self.professions_list = read_professions_file(professions_file)

        if not self.professions_list:
            print("[WARNING]: No professions list specified. Will use the single profession from settings.json.")
            self.professions_list = [self.settings.options.get("text")]

        print(f"[INFO]: Will process {len(self.professions_list)} professions:")
        for i, prof in enumerate(self.professions_list, 1):
            print(f"  {i}. {prof}")

        self.exchanger = Exchanger(config_path)
        self.collector = None
        self.analyzer = None
        self.predictor = Predictor()

    def process_all(self):
        """Process all professions in the list."""
        # Update exchange rates once for all professions
        if not any(self.settings.rates.values()) or self.settings.update:
            print("[INFO]: Trying to get exchange rates from remote server...")
            self.exchanger.update_exchange_rates(self.settings.rates)
            self.exchanger.save_rates(self.settings.rates)

        print(f"[INFO]: Exchange rates: {self.settings.rates}")
        self.collector = DataCollector(self.settings.rates)

        # Create base data directory if it doesn't exist
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
            print(f"[INFO]: Created base directory: {self.base_dir}")

        # Process each profession
        for i, profession in enumerate(self.professions_list, 1):
            print(f"\n{'='*80}")
            print(f"[INFO]: Processing profession {i}/{len(self.professions_list)}: {profession}")
            print(f"{'='*80}")

            # Create safe directory name for this profession
            safe_profession = profession.replace(" ", "_").replace("/", "_").replace("\\", "_")
            profession_dir = os.path.join(self.base_dir, safe_profession)

            # Create directory for this profession
            if not os.path.exists(profession_dir):
                os.makedirs(profession_dir)
                print(f"[INFO]: Created directory for profession: {profession_dir}")

            # Create analyzer with the correct output directory
            self.analyzer = Analyzer(self.settings.save_result)

            # Update the text field in options
            original_text = self.settings.options["text"]
            self.settings.options["text"] = profession

            try:
                # Collect data for this profession
                vacancies = self.collector.collect_vacancies(
                    query=self.settings.options, refresh=self.settings.refresh, num_workers=self.settings.num_workers
                )

                # Check if we got any results
                if not vacancies or not any(vacancies.values()):
                    print(f"[WARNING]: No vacancies found for '{profession}', skipping analysis.")
                    continue

                # Prepare and analyze dataframe
                df = self.analyzer.prepare_df(vacancies)
                if df.empty:
                    print(f"[WARNING]: No data to analyze for '{profession}', skipping.")
                    continue

                print("\n[INFO]: Analyzing dataframe...")
                self.analyzer.analyze_df(df)

                # Save results in the profession-specific directory
                output_filename = os.path.join(profession_dir, "salary_analysis.csv")
                df_csv_filename = os.path.join(profession_dir, "hh_results.csv")

                if self.settings.save_result:
                    print(f"[INFO]: Saving DataFrame to {df_csv_filename}")
                    df.to_csv(df_csv_filename, index=False)

                self.analyzer.analyze_and_save_results(df, output_filename=output_filename)

                print(f"[INFO]: Done with profession: {profession}!")
                print(f"[INFO]: Results saved in: {profession_dir}")

            except Exception as e:
                print(f"[ERROR]: Failed to process profession '{profession}': {str(e)}")
                import traceback

                traceback.print_exc()
            finally:
                # Restore original text in options
                self.settings.options["text"] = original_text

        print("\n[INFO]: All professions processing completed!")


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Example usage:
    # 1. Use professions from professions.txt:
    # python multi_processor.py
    #
    # 2. Or provide a custom list via command line:
    # python multi_processor.py --professions "Data Scientist" "Python Developer"
    #
    # 3. Specify a different professions file:
    # python multi_processor.py --professions_file my_professions.txt

    processor = MultiProcessor(
        config_path=args.config,
        professions_list=args.professions,
        professions_file=args.professions_file,
        base_dir=args.base_dir,
    )
    processor.process_all()
