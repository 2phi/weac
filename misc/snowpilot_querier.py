# Standard library imports
import os
import shutil
import calendar
import tarfile
from datetime import datetime, timedelta
from pathlib import Path
from glob import glob
from time import sleep
import logging

# Third-party imports
import requests
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv(override=True)

# Set up logging
logger = logging.getLogger(__name__)


class SnowPilotQuerier:
    """
    A class to query the SnowPilot API for CAAML data organized by year.

    This class provides methods to query the SnowPilot API, download snow pit
    observations in CAAML format for entire years, and manage data with
    intelligent caching organized by year.

    Parameters
    ----------
    data_path : str or Path, optional
        The path to the data directory. Default is 'data/snowpilot'.
    caaml_path : str or Path, optional
        The path to the CAAML directory. Default is 'data/snowpilot/caaml'.

    Attributes
    ----------
    data_path : Path
        The path to the data directory.
    caaml_path : Path
        The path to the CAAML directory.
    site_url : str
        The URL of the SnowPilot website.
    log_in_url : str
        The URL for user login to the SnowPilot website.
    caaml_query_url : str
        The URL for querying CAAML data.
    data_url : str
        The URL for downloading data.
    credentials : dict
        The login credentials for the SnowPilot website.
    """

    def __init__(
        self,
        data_path: str | Path = "data/snowpilot",
        caaml_path: str | Path = None,
    ) -> None:
        # Directories
        self.data_path = Path(data_path)
        self.caaml_path = Path(caaml_path) if caaml_path else self.data_path / "caaml"

        # Create directories if they don't exist
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.caaml_path.mkdir(parents=True, exist_ok=True)

        # URLs
        self.site_url = "https://snowpilot.org"
        self.log_in_url = self.site_url + "/user/login"
        self.caaml_query_url = self.site_url + "/avscience-query-caaml.xml?"
        self.data_url = "https://snowpilot.org/sites/default/files/tmp/"

        # Login credentials
        self.credentials = {
            "name": os.environ.get("SNOWPILOT_USER"),
            "pass": os.environ.get("SNOWPILOT_PASSWORD"),
            "form_id": "user_login",
            "op": "Log in",
        }

        if not self.credentials["name"] or not self.credentials["pass"]:
            logger.warning("SnowPilot credentials not found in environment variables")

    def query_year(self, year: int, force_download: bool = False) -> bool:
        """
        Query SnowPilot for a complete year of data.

        Parameters
        ----------
        year : int
            Year to download (e.g., 2023).
        force_download : bool, optional
            If True, download even if data already exists. Default is False.

        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        # Create year directory
        year_path = self.caaml_path / str(year)
        year_path.mkdir(exist_ok=True)

        # Check if data already exists
        tar_filename = f"{year}.tar.gz"
        tar_path = year_path / tar_filename

        if tar_path.exists() and not force_download:
            logger.info("Data for year %d already exists, skipping download", year)
            return True

        # Check if extracted CAAML files already exist
        existing_caaml = list(year_path.glob("*.caaml"))
        if existing_caaml and not force_download:
            logger.info(
                "Extracted CAAML files for year %d already exist, skipping download",
                year,
            )
            return True

        # Define date range for the year
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        logger.info("Downloading data for year %d", year)
        success, message = self._download_caaml(start_date, end_date, year)

        if success:
            logger.info("Successfully downloaded data: %s", message)
            self._extract_caaml_files([tar_path], year)
            return True
        else:
            logger.error("Failed to download data: %s", message)
            return False

    def query_years(
        self, years: list, pause_between: int = 10, force_download: bool = False
    ) -> dict:
        """
        Query SnowPilot for multiple years of data.

        Parameters
        ----------
        years : list of int
            List of years to download (e.g., [2022, 2023, 2024]).
        pause_between : int, optional
            Seconds to pause between downloads. Default is 10.
        force_download : bool, optional
            If True, download even if data already exists. Default is False.

        Returns
        -------
        dict
            Dictionary with results for each year.
        """
        results = {}

        with tqdm(total=len(years), desc="Querying SnowPilot") as pbar:
            for year in years:
                pbar.set_postfix({"Year": year})

                result = self.query_year(year, force_download)
                results[year] = result

                pbar.update(1)

                if pause_between > 0:
                    sleep(pause_between)

        return results

    def get_available_data(self) -> dict:
        """
        Get list of available CAAML data files organized by year.

        Returns
        -------
        dict
            Dictionary with available data organized by year.
        """
        available_years = {}

        # Check each year subdirectory
        for year_dir in self.caaml_path.iterdir():
            if year_dir.is_dir() and year_dir.name.isdigit():
                year = int(year_dir.name)

                tar_files = list(year_dir.glob("*.tar.gz"))
                caaml_files = list(year_dir.glob("*.caaml"))

                available_years[year] = {
                    "year_path": year_dir,
                    "compressed_files": [
                        {"file": f.name, "path": f} for f in tar_files
                    ],
                    "caaml_files": [{"file": f.name, "path": f} for f in caaml_files],
                    "has_compressed": len(tar_files) > 0,
                    "has_extracted": len(caaml_files) > 0,
                }

        return available_years

    def extract_all_caaml(self) -> None:
        """Extract all .tar.gz files to individual .caaml files."""
        total_files = 0

        # Check each year subdirectory
        for year_dir in self.caaml_path.iterdir():
            if year_dir.is_dir() and year_dir.name.isdigit():
                year = int(year_dir.name)
                tar_files = list(year_dir.glob("*.tar.gz"))

                if tar_files:
                    logger.info("Extracting %d files for year %d", len(tar_files), year)
                    self._extract_caaml_files(tar_files, year)
                    total_files += len(tar_files)

        if total_files == 0:
            logger.info("No compressed files found to extract")
        else:
            logger.info("Extracted %d total compressed files", total_files)

    def _download_caaml(self, start_date: str, end_date: str, year: int) -> tuple:
        """
        Download CAAML data for a given date range.

        Parameters
        ----------
        start_date : str
            Start date in YYYY-MM-DD format.
        end_date : str
            End date in YYYY-MM-DD format.
        year : int
            Year for organizing the data.

        Returns
        -------
        tuple
            (success: bool, message: str)
        """
        # Query string
        query = f"OBS_DATE_MIN={start_date}&OBS_DATE_MAX={end_date}&per_page=1000"

        try:
            with requests.Session() as session:
                # Authenticate
                auth_response = session.post(self.log_in_url, data=self.credentials)
                if auth_response.status_code != 200:
                    return False, "Authentication failed"

                # Query CAAML feed
                response = session.post(self.caaml_query_url + query)

                if response.status_code != 200:
                    return False, f"Query failed with status {response.status_code}"

                # Get content disposition to find the file
                disposition = response.headers.get("Content-Disposition", "")

                if len(disposition) < 40:
                    return False, "No data found for this date range"

                # Extract filename and download the data file
                filename = disposition[22:-1].replace("_caaml", "")
                file_url = self.data_url + filename

                data_response = session.get(file_url)
                if data_response.status_code != 200:
                    return (
                        False,
                        f"Data download failed with status {data_response.status_code}",
                    )

                # Save the compressed file in year directory
                year_path = self.caaml_path / str(year)
                save_filename = f"{year}.tar.gz"
                save_path = year_path / save_filename

                with open(save_path, "wb") as f:
                    f.write(data_response.content)

                return True, f"Downloaded {save_filename}"

        except Exception as e:
            return False, f"Download error: {str(e)}"

    def _extract_caaml_files(self, tar_files: list, year: int) -> None:
        """
        Extract CAAML files from tar.gz archives to year directory.

        Parameters
        ----------
        tar_files : list
            List of tar.gz file paths to extract.
        year : int
            Year for organizing the extracted files.
        """
        year_path = self.caaml_path / str(year)

        for tar_path in tar_files:
            try:
                with tarfile.open(tar_path, "r:gz") as tar:
                    # Extract all .caaml files
                    for member in tar.getmembers():
                        if member.name.endswith(".caaml") or member.name.endswith(
                            "caaml.xml"
                        ):
                            # Extract to a temporary location first
                            tar.extract(member, path=year_path)

                            # Move the file to the year directory with a clean name
                            extracted_path = year_path / member.name
                            if extracted_path.exists():
                                # Create a clean filename
                                clean_name = Path(member.name).name
                                if not clean_name.endswith(".caaml"):
                                    clean_name = clean_name.replace(".xml", ".caaml")

                                final_path = year_path / clean_name

                                # Handle duplicate names by adding a counter
                                counter = 1
                                while final_path.exists():
                                    stem = final_path.stem
                                    suffix = final_path.suffix
                                    final_path = year_path / f"{stem}_{counter}{suffix}"
                                    counter += 1

                                shutil.move(str(extracted_path), str(final_path))

                                # Clean up any intermediate directories
                                parent_dir = extracted_path.parent
                                if parent_dir != year_path and parent_dir.exists():
                                    try:
                                        shutil.rmtree(parent_dir)
                                    except OSError:
                                        pass  # Directory might not be empty

                logger.info(
                    "Extracted CAAML files from %s to year %d", tar_path.name, year
                )

            except Exception as e:
                logger.error("Failed to extract %s: %s", tar_path.name, str(e))

    def cleanup_compressed_files(self, year: int = None) -> None:
        """
        Remove .tar.gz files after extraction to save space.

        Parameters
        ----------
        year : int, optional
            Specific year to clean up. If None, cleans all years.
        """
        total_removed = 0

        if year is not None:
            # Clean up specific year
            year_path = self.caaml_path / str(year)
            if year_path.exists():
                tar_files = list(year_path.glob("*.tar.gz"))
                for tar_file in tar_files:
                    try:
                        tar_file.unlink()
                        logger.info("Removed compressed file: %s", tar_file.name)
                        total_removed += 1
                    except OSError as e:
                        logger.error("Failed to remove %s: %s", tar_file.name, str(e))
        else:
            # Clean up all years
            for year_dir in self.caaml_path.iterdir():
                if year_dir.is_dir() and year_dir.name.isdigit():
                    tar_files = list(year_dir.glob("*.tar.gz"))
                    for tar_file in tar_files:
                        try:
                            tar_file.unlink()
                            logger.info("Removed compressed file: %s", tar_file.name)
                            total_removed += 1
                        except OSError as e:
                            logger.error(
                                "Failed to remove %s: %s", tar_file.name, str(e)
                            )

        logger.info("Removed %d compressed files", total_removed)


if __name__ == "__main__":
    querier = SnowPilotQuerier()
    querier.query_year(2024)
