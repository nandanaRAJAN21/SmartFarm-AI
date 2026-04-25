"""
Data loader module for loading crop yield and fertilizer datasets.
"""

import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """Class to handle loading and saving of datasets."""
    
    BASE_PATH = r"C:\Users\Nowshad\Desktop\Antigravity_projects\crop yield prediction\datasets"

    def __init__(self):
        """Initialize DataLoader."""
        self.base_path = Path(self.BASE_PATH)

    def load_yield(self) -> pd.DataFrame:
        """Load crop_yield_dataset.csv."""
        file_path = self.base_path / "crop_yield_dataset.csv"
        try:
            logger.info(f"Loading yield dataset from {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded yield dataset. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading yield dataset: {e}")
            raise

    def load_fertilizer(self) -> pd.DataFrame:
        """Load fertilizer_recommendation.csv."""
        file_path = self.base_path / "fertilizer_recommendation.csv"
        try:
            logger.info(f"Loading fertilizer dataset from {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded fertilizer dataset. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading fertilizer dataset: {e}")
            raise

    def get_info(self, df: pd.DataFrame) -> None:
        """Print dataset info, dtypes, nulls, and describe."""
        print("=" * 50)
        print("DATASET INFO")
        print("=" * 50)
        print(f"Shape: {df.shape}\n")
        print("Data Types:")
        print(df.dtypes)
        print("\nNull Values:")
        print(df.isnull().sum())
        print("\nStatistical Summary:")
        print(df.describe(include='all'))
        
    def get_sample(self, df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """Return the first n rows of the dataframe."""
        return df.head(n)

    def save_processed(self, df: pd.DataFrame, fname: str) -> None:
        """Save dataframe to data/processed/ directory."""
        save_path = Path(r"C:\Users\Nowshad\Desktop\Antigravity_projects\crop yield prediction\data\processed") / fname
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving processed data to {save_path}")
            df.to_csv(save_path, index=False)
            logger.info("Successfully saved processed data.")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise
