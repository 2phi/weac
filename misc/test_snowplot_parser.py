#!/usr/bin/env python3
"""
Simple script to extract and print all values from the CAAML file for analysis.
"""

from weac_2.utils.snowpilot_parser import SnowPilotParser


def analyze_caaml_file(file_path: str):
    """Extract and print all values from the CAAML file."""
    print(f"Analyzing CAAML file: {file_path}")
    print("=" * 60)

    # Parse the file
    snowpit_parser = SnowPilotParser(file_path)
    model_inputs = snowpit_parser.run()

    # Print snowpit basic info
    snowpit = snowpit_parser.snowpit
    print("\n📍 LOCATION & BASIC INFO:")
    print(f"  Location: {snowpit.core_info.location}")
    print(f"  Elevation: {snowpit.core_info.location.elevation}")
    print(f"  Aspect: {snowpit.core_info.location.aspect}")
    print(f"  Slope angle: {snowpit.core_info.location.slope_angle}")
    print(f"  Profile depth: {snowpit.snow_profile.profile_depth}")

    # Print extracted layers
    print("\n🏔️  EXTRACTED LAYERS:")
    print("  Layer | Depth Top | Thickness | Density | Grain Form | Hardness")
    print("  ------|-----------|-----------|---------|------------|----------")

    total_depth = 0
    for i, layer in enumerate(snowpit_parser.layers, 1):
        # Get original snowpylot layer for additional info
        sp_layer = None
        current_depth = 0
        for sp_l in snowpit.snow_profile.layers:
            if sp_l.depth_top is not None:
                if current_depth == total_depth:
                    sp_layer = sp_l
                    break
                current_depth += (
                    sp_l.thickness[0] * 10 if sp_l.thickness else 0
                )  # Convert to mm

        depth_top_cm = total_depth / 10  # Convert mm to cm for display
        thickness_cm = layer.h / 10  # Convert mm to cm for display

        grain_form = "N/A"
        hardness = "N/A"
        if sp_layer:
            if sp_layer.grain_form_primary and sp_layer.grain_form_primary.grain_form:
                grain_form = sp_layer.grain_form_primary.grain_form
            if sp_layer.hardness:
                hardness = sp_layer.hardness
            elif sp_layer.hardness_top and sp_layer.hardness_bottom:
                hardness = f"{sp_layer.hardness_top}-{sp_layer.hardness_bottom}"

        print(
            f"  {i:5d} | {depth_top_cm:9.1f} | {thickness_cm:9.1f} | {layer.rho:7.1f} | {grain_form:10s} | {hardness}"
        )
        total_depth += layer.h

    print(f"\n  Total depth: {total_depth / 10:.1f} cm ({total_depth:.0f} mm)")

    # Print stability tests
    print("\n🧪 STABILITY TESTS:")

    # PST tests
    psts = snowpit.stability_tests.PST
    if psts:
        print(f"  PST Tests: {len(psts)}")
        for i, pst in enumerate(psts, 1):
            print(
                f"    PST {i}: depth_top={pst.depth_top}, cut_length={pst.cut_length}, column_length={pst.column_length}"
            )
    else:
        print("  PST Tests: None")

    # ECT tests
    ects = snowpit.stability_tests.ECT
    if ects:
        print(f"  ECT Tests: {len(ects)}")
        for i, ect in enumerate(ects, 1):
            depth_mm = (
                ect.depth_top[0] * 10 if ect.depth_top else "N/A"
            )  # Convert to mm
            print(f"    ECT {i}: depth_top={ect.depth_top} ({depth_mm} mm)")
    else:
        print("  ECT Tests: None")

    # CT tests
    cts = snowpit.stability_tests.CT
    if cts:
        print(f"  CT Tests: {len(cts)}")
        for i, ct in enumerate(cts, 1):
            depth_mm = ct.depth_top[0] * 10 if ct.depth_top else "N/A"  # Convert to mm
            print(f"    CT {i}: depth_top={ct.depth_top} ({depth_mm} mm)")
    else:
        print("  CT Tests: None")

    # RBlock tests
    rblocks = snowpit.stability_tests.RBlock
    if rblocks:
        print(f"  RBlock Tests: {len(rblocks)}")
        for i, rb in enumerate(rblocks, 1):
            depth_mm = rb.depth_top[0] * 10 if rb.depth_top else "N/A"  # Convert to mm
            print(f"    RBlock {i}: depth_top={rb.depth_top} ({depth_mm} mm)")
    else:
        print("  RBlock Tests: None")

    # Print weak layer analysis for stability test depths
    print("\n🎯 WEAK LAYER ANALYSIS:")

    # Collect all test depths
    test_depths = set()
    for ect in ects:
        if ect.depth_top:
            test_depths.add(ect.depth_top[0] * 10)  # Convert to mm
    for ct in cts:
        if ct.depth_top:
            test_depths.add(ct.depth_top[0] * 10)  # Convert to mm
    for rb in rblocks:
        if rb.depth_top:
            test_depths.add(rb.depth_top[0] * 10)  # Convert to mm

    if test_depths:
        for depth_mm in sorted(test_depths):
            print(f"\n  At depth {depth_mm} mm ({depth_mm / 10} cm):")
            try:
                weak_layer, layers_above = (
                    snowpit_parser._extract_weak_layer_and_layers_above(
                        snowpit, depth_mm, snowpit_parser.layers
                    )
                )

                print(
                    f"    Weak layer: density={weak_layer.rho:.1f} kg/m³, thickness={weak_layer.h:.1f} mm"
                )
                print(f"    Layers above ({len(layers_above)}):")

                for i, layer in enumerate(layers_above, 1):
                    print(
                        f"      Layer {i}: thickness={layer.h:.1f} mm, density={layer.rho:.1f} kg/m³"
                    )

                total_above = sum(layer.h for layer in layers_above)
                print(
                    f"    Total depth above weak layer: {total_above:.1f} mm ({total_above / 10:.1f} cm)"
                )

            except Exception as e:
                print(f"    Error extracting weak layer: {e}")
    else:
        print("  No stability test depths found")

    # Print model inputs
    print("\n📊 GENERATED MODEL INPUTS:")
    model_inputs = snowpit_parser.get_model_inputs()
    print(f"  Number of scenarios: {len(model_inputs)}")

    for i, model_input in enumerate(model_inputs, 1):
        print(f"\n  Scenario {i}:")
        print(f"    System type: {model_input.scenario_config.system_type}")
        print(f"    Slope angle: {model_input.scenario_config.phi}°")
        print(f"    Layers above weak layer: {len(model_input.layers)}")

        total_depth_above = sum(layer.h for layer in model_input.layers)
        print(
            f"    Total depth above: {total_depth_above:.1f} mm ({total_depth_above / 10:.1f} cm)"
        )
        print(
            f"    Weak layer: density={model_input.weak_layer.rho:.1f} kg/m³, thickness={model_input.weak_layer.h:.1f} mm"
        )
        print(f"    Segments: {len(model_input.segments)}")

        for j, segment in enumerate(model_input.segments, 1):
            print(
                f"      Segment {j}: length={segment.length} mm, foundation={segment.has_foundation}"
            )


if __name__ == "__main__":
    # analyze_caaml_file("data/Cairn Gully-10-Jun.caaml")
    # analyze_caaml_file("data/Hatcher, prez ridge-02-Apr.caaml")
    # analyze_caaml_file("data/Windluck-09-Apr.caaml")
    # analyze_caaml_file("data/Ellis upper elevation-13-Mar.caaml")
    analyze_caaml_file("data/Falsa Parva-10-Jul.caaml")
