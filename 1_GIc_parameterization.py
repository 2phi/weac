import os
from typing import List
import pandas as pd
from pprint import pprint

from weac_2.analysis import Analyzer
from weac_2.core.system_model import SystemModel
from weac_2.components import ModelInput, Segment, ScenarioConfig
from weac_2.utils.snowpilot_parser import SnowPilotParser, convert_to_mm


# Process multiple files
file_paths = []
for directory in os.listdir("data/snowpits"):
    for file in os.listdir(f"data/snowpits/{directory}"):
        if file.endswith(".xml"):
            file_paths.append(f"data/snowpits/{directory}/{file}")

pst_paths: List[str] = []
pst_parsers: List[SnowPilotParser] = []
for file_path in file_paths:
    snowpilot_parser = SnowPilotParser(file_path)
    if len(snowpilot_parser.snowpit.stability_tests.PST) > 0:
        pst_paths.append(file_path)
        pst_parsers.append(snowpilot_parser)

print(f"\nFound {len(pst_paths)} files with PST tests")

# Extract data from all PST files
error_paths = {}
error_values = {}

# dataframe = pd.DataFrame(
#     columns=[
#         "file_path",
#         "column_length",
#         "cut_length",
#         "cut_depth",
#         "layers",
#     ]
# )
for i, (file_path, parser) in enumerate(zip(pst_paths, pst_parsers)):
    try:
        phi = parser.snowpit.core_info.location.slope_angle
        layers = parser.extract_layers()
        for pst in parser.snowpit.stability_tests.PST:
            weak_layer, layers_above = parser.extract_weak_layer_and_layers_above(
                pst.depth_top[0] * convert_to_mm[pst.depth_top[1]], layers
            )
            print(layers)
            print(weak_layer)
            print(layers_above)
    except Exception as e:
        print(e)
        error_paths[i] = file_path
        error_values[i] = e
print(len(error_paths))
print(len(error_values))
pprint(error_paths)
pprint(error_values)
breakpoint()
# dataframe = dataframe.append(
#     {
#         "file_path": file_path,
#         "column_length": pst.column_length,
#         "cut_length": pst.cut_length,
#         "cut_depth": pst.depth_top,
#         "layers": layers_above,
#         "weak_layer": weak_layer,
#     },
# )
# segments = [
#     Segment(length=pst.cut_length, found_depth=False, m=0.0),
#     Segment(
#         length=pst.column_length - pst.cut_length,
#         found_depth=True,
#         m=0.0,
#     ),
# ]
# scenario_config = ScenarioConfig(system_type="-pst", phi=phi)
# model_input = ModelInput(
#     weak_layer=weak_layer,
#     layers=layers_above,
#     scenario_config=scenario_config,
#     segments=segments,
# )
# pst_system = SystemModel(model_input=model_input)
# pst_analyzer = Analyzer(pst_system)
# G, GIc, GIIc = pst_analyzer.differential_ERR(unit="J/m^2")
# print(G, GIc, GIIc)
