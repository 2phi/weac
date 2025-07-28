import os
from typing import List
from numpy.linalg import LinAlgError
import pandas as pd
from pprint import pprint
import tqdm

from weac_2.analysis import Analyzer
from weac_2.core.system_model import SystemModel
from weac_2.components import ModelInput, Segment, ScenarioConfig
from weac_2.utils.snowpilot_parser import SnowPilotParser, convert_to_mm, convert_to_deg


# Process multiple files
file_paths = []
for directory in os.listdir("data/snowpits"):
    for file in os.listdir(f"data/snowpits/{directory}"):
        if file.endswith(".xml"):
            file_paths.append(f"data/snowpits/{directory}/{file}")

pst_paths: List[str] = []
pst_parsers: List[SnowPilotParser] = []
amount_of_psts = 0

for file_path in file_paths:
    snowpilot_parser = SnowPilotParser(file_path)
    if len(snowpilot_parser.snowpit.stability_tests.PST) > 0:
        pst_paths.append(file_path)
        pst_parsers.append(snowpilot_parser)
        amount_of_psts += len(snowpilot_parser.snowpit.stability_tests.PST)

print(f"\nFound {len(pst_paths)} files with PST tests")
print(f"Found {amount_of_psts} PST tests")

# Extract data from all PST files
error_paths = {}
error_values = {}
failed_to_extract_layers = 0
overall_excluded_psts = 0
cut_length_exceeds_column_length = 0
slope_angle_is_None = 0
failed_to_extract_weak_layer = 0

data_rows = []
for i, (file_path, parser) in tqdm.tqdm(
    enumerate(zip(pst_paths, pst_parsers)), total=len(pst_paths)
):
    try:
        if parser.snowpit.core_info.location.slope_angle is None:
            phi = 0.0
        else:
            phi = (
                parser.snowpit.core_info.location.slope_angle[0]
                * convert_to_deg[parser.snowpit.core_info.location.slope_angle[1]]
            )
        try:
            layers, density_method = parser.extract_layers()
            if density_method == "density_obs":
                print(f"Density method: {density_method}")
                breakpoint()
        except Exception as e:
            failed_to_extract_layers += len(parser.snowpit.stability_tests.PST)
            raise e
        for pst_id, pst in enumerate(parser.snowpit.stability_tests.PST):
            try:
                if pst.cut_length[0] >= pst.column_length[0]:
                    cut_length_exceeds_column_length += 1
                    raise ValueError(
                        "Cut length is equal or greater than column length"
                    )
                try:
                    weak_layer, layers_above = (
                        parser.extract_weak_layer_and_layers_above(
                            pst.depth_top[0] * convert_to_mm[pst.depth_top[1]], layers
                        )
                    )
                except Exception as e:
                    failed_to_extract_weak_layer += 1
                    raise e
                cut_length = pst.cut_length[0] * convert_to_mm[pst.cut_length[1]]
                column_length = (
                    pst.column_length[0] * convert_to_mm[pst.column_length[1]]
                )
                segments = [
                    Segment(length=cut_length, has_foundation=False, m=0.0),
                    Segment(
                        length=column_length - cut_length,
                        has_foundation=True,
                        m=0.0,
                    ),
                ]
                scenario_config = ScenarioConfig(system_type="-pst", phi=phi)
                model_input = ModelInput(
                    weak_layer=weak_layer,
                    layers=layers_above,
                    scenario_config=scenario_config,
                    segments=segments,
                )
                pst_system = SystemModel(model_input=model_input)
                pst_analyzer = Analyzer(pst_system)
                G, GIc, GIIc = pst_analyzer.differential_ERR(unit="J/m^2")

                data_rows.append(
                    {
                        "file_path": file_path,
                        "pst_id": pst_id,
                        "column_length": column_length,
                        "cut_length": cut_length,
                        "phi": phi,
                        # Weak Layer properties
                        "rho_wl": weak_layer.rho,
                        "E_wl": weak_layer.E,
                        "HH_wl": weak_layer.hand_hardness,
                        "GT_wl": weak_layer.grain_type,
                        "GS_wl": weak_layer.grain_size,
                        # Simulation results
                        "G": G,
                        "GIc": GIc,
                        "GIIc": GIIc,
                    }
                )
            except Exception as e:
                error_id = f"{i}.{pst_id}"
                error_paths[error_id] = file_path
                error_values[error_id] = e
                overall_excluded_psts += 1

    except Exception as e:
        error_values[str(i)] = e
        error_paths[str(i)] = file_path
        overall_excluded_psts += len(parser.snowpit.stability_tests.PST)

dataframe = pd.DataFrame(data_rows)
pprint(error_values)
print(f"\nFound {len(pst_paths)} files with PST tests")
print(f"Found {amount_of_psts} PST tests")
print("Length of the dataframe: ", len(dataframe))
print(f"Amount of excluded PSTs: {overall_excluded_psts}")

print(f"\nFailed to extract layers: {failed_to_extract_layers}")
print(f"Failed to extract weak layer: {failed_to_extract_weak_layer}")
print(f"Slope angle is None: {slope_angle_is_None}")
print(f"Cut length exceeds column length: {cut_length_exceeds_column_length}")
print(
    f"Added Failure Types: {failed_to_extract_layers + slope_angle_is_None + cut_length_exceeds_column_length + failed_to_extract_weak_layer}"
)

# exclude dataframes where the cut_length is greater than 60% of the column length
if not dataframe.empty:
    dataframe = dataframe[dataframe["cut_length"] < 0.6 * dataframe["column_length"]]
    print("Length of the dataframe after exclusion: ", len(dataframe))
    print(dataframe.head())

# # Save the data to a csv file
dataframe.to_csv("pst_to_GIc.csv", index=False)
