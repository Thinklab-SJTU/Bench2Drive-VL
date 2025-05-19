import xml.etree.ElementTree as ET
import json

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def parse_scenarios(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    scenario_data = {}
    filtered_scenario_data = {}

    for route in root.findall("route"):
        for scenario in route.findall("scenarios/scenario"):
            scenario_type = scenario.get("type")
            if scenario_type not in scenario_data:
                scenario_data[scenario_type] = {}
                filtered_scenario_data[scenario_type] = {}

            for element in scenario:
                tag = element.tag
                for key, value in element.attrib.items():
                    if tag not in scenario_data[scenario_type]:
                        scenario_data[scenario_type][tag] = {}
                        if key != "trigger_point" and tag != "trigger_point":
                            filtered_scenario_data[scenario_type][tag] = {}

                    if key not in scenario_data[scenario_type][tag]:
                        scenario_data[scenario_type][tag][key] = []
                        if key != "trigger_point" and tag != "trigger_point":
                            filtered_scenario_data[scenario_type][tag][key] = []

                    if value not in scenario_data[scenario_type][tag][key]:
                        scenario_data[scenario_type][tag][key].append(value)
                    
                    if not is_number(value) and key != "trigger_point" and tag != "trigger_point" and value not in filtered_scenario_data[scenario_type][tag][key]:
                        filtered_scenario_data[scenario_type][tag][key].append(value)

    sorted_scenario_data = {k: scenario_data[k] for k in sorted(scenario_data)}
    sorted_filtered_scenario_data = {k: filtered_scenario_data[k] for k in sorted(filtered_scenario_data)}

    with open("scenario_statistics.json", "w", encoding="utf-8") as f:
        json.dump(sorted_scenario_data, f, indent=4, ensure_ascii=False)

    with open("scenario_statistics_filtered.json", "w", encoding="utf-8") as f:
        json.dump(sorted_filtered_scenario_data, f, indent=4, ensure_ascii=False)

    return sorted_scenario_data, sorted_filtered_scenario_data

if __name__ == "__main__":
    xml_file = "./leaderboard/data/bench2drive220.xml"
    result, filtered_result = parse_scenarios(xml_file)
