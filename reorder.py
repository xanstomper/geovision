with open('geovision_deep_scan.py', 'r') as f:
    content = f.read()

# We want to move Phase 8 to run after Phase 4.

import re

phase4_regex = r"(    # Phase 4b: Property Records\n.*?)(    # Phase 9: Synthesis\n)"
match = re.search(phase4_regex, content, re.DOTALL)
if match:
    block = match.group(1)
    
    # Split the block into sections:
    # 4b, 5, 6, 7, 8
    p4b_split = block.split("    # Phase 5: Satellite Matching\n")
    p4b = p4b_split[0]
    
    p5_split = p4b_split[1].split("    # Phase 6: Park Proximity\n")
    p5 = "    # Phase 5: Satellite Matching\n" + p5_split[0]
    
    p6_split = p5_split[1].split("    # Phase 7: Cross-View Verification\n")
    p6 = "    # Phase 6: Park Proximity\n" + p6_split[0]
    
    p7_split = p6_split[1].split("    # Phase 8: VLM Analysis (with context)\n")
    p7 = "    # Phase 7: Cross-View Verification\n" + p7_split[0]
    
    p8 = "    # Phase 8: VLM Analysis (with context)\n" + p7_split[1]
    
    # Now modify Phase 8 context payload since property_records, satellite_matches won't exist yet!
    p8_new = p8.replace('"property_records": pipeline_result.property_records.get("records", [])[:3],', '# property records not yet available')
    p8_new = p8_new.replace('"satellite_matches": pipeline_result.satellite_matches.get("matches", [])[:3]', '# satellite matches not yet available')

    # Now we want the order: 8, 4b, 5, 6, 7
    # But wait, we need to inject the VLM matches into db_matches so that 4b, 5, 6, 7 can use them!
    vlm_injection = """
    # Inject VLM estimates into db_matches for subsequent phases to use
    if pipeline_result.vlm_analysis.get("location_estimates"):
        for est in pipeline_result.vlm_analysis["location_estimates"]:
            pipeline_result.db_matches.setdefault("matches", []).insert(0, {
                "city": "VLM Estimate",
                "latitude": est["latitude"],
                "longitude": est["longitude"],
                "confidence": est["confidence"],
                "matched_feature": "VLM Intelligence"
            })
"""

    new_block = p8_new + vlm_injection + p4b + p5 + p6 + p7
    
    content = content[:match.start()] + new_block + "    # Phase 9: Synthesis\n" + content[match.end():]
    
    with open('geovision_deep_scan.py', 'w') as f:
        f.write(content)
    print("Reordered successfully!")
else:
    print("Regex failed to match!")
