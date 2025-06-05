"""
This module defines the parser for the Snowpilot/Snowpack data.
The parser is used to parse the Snowpilot/Snowpack data into a format that can be used by the WEAC simulation.
"""
import logging
from typing import Literal, Optional
from weac_2.components.model_input import ModelInput

logger = logging.getLogger(__name__)


class SnowprofileParser:
    """
    This class is used to parse the Snowpilot/Snowpack data into a format that can be used by the WEAC simulation.
    """
    format: Literal["snowpilot", "snowpack"] = "snowpilot"
    file_path: Optional[str] = None
    data: Optional[str] = None
    model_input: ModelInput = ModelInput()

    def parse(self, format: Literal["snowpilot", "snowpack"], file_path: Optional[str] = None, data: Optional[str] = None):
        # Set the format
        self.format = format
        # Set the file path
        self.file_path = file_path
        # Set the data
        self.data = data
        # Parse the data
        if self.format == "snowpilot":
            self._parse_snowpilot()
        elif self.format == "snowpack":
            self._parse_snowpack()
        else:
            raise ValueError(f"Invalid format: {self.format}")
        return self.model_input

    def _parse_snowpilot(self):
        if self.file_path is not None:
            with open(self.file_path, "r") as file:
                self.data = file.read()
        elif self.data is not None:
            self.data = self.data
        # TODO: Cast Snowpilot data to ModelInput
    
    def _parse_snowpack(self):
        if self.file_path is not None:
            with open(self.file_path, "r") as file:
                self.data = file.read()
        elif self.data is not None:
            self.data = self.data
        # TODO: Cast Snowpack data to ModelInput