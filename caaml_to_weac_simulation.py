import logging

from weac_2.logging_config import setup_logging
from weac_2.utils.CAAML_to_weac import convert_snowpit_to_weac

setup_logging(level="INFO")

logger = logging.getLogger(__name__)


file_path = "Cairn Gully-10-Jun.caaml"
model_inputs = convert_snowpit_to_weac(file_path)

for model_input in model_inputs:
    print(model_input)
