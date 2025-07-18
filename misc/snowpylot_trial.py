from snowpylot import caaml_parser
from snowpylot.snow_pit import SnowPit

# Parse a CAAML file
snowpit: SnowPit = caaml_parser(
    "/home/pillowbeast/Documents/weac/misc/Cairn Gully-10-Jun.caaml"
)

print(f"Snowpit: {snowpit}")
print(f"Core Info: {snowpit.core_info}")
print(f"Snow Profile: {snowpit.snow_profile}")
print(f"Stability Tests: {snowpit.stability_tests}")
print(f"Whumpf Data: {snowpit.whumpf_data}")

with open("snowpit.txt", "w") as f:
    f.write(str(snowpit))

# # Access basic information
# print(f"Pit ID: {snowpit.core_info.pit_id}")
# print(f"Date: {snowpit.core_info.date}")
# print(f"Location: {snowpit.core_info.location.latitude}, {snowpit.core_info.location.longitude}")

# # Access snow profile data
# print(f"HS: {snowpit.snow_profile.hs}")

# # Access layer information
# for i, layer in enumerate(snowpit.snow_profile.layers):
#     print(f"Layer {i+1}: Depth {layer.depth_top}, Thickness {layer.thickness}")
#     print(f"  Grain form: {layer.grain_form_primary.grain_form}")
#     print(f"  Hardness: {layer.hardness}")

# # Access ECT test results
# for ect in snowpit.stability_tests.ECT:
#     print(f"ECT at depth {ect.depth_top}: Score {ect.test_score}")
