import json


with open('OneForAll/infer_json_vehicle.json', 'r') as f:
    result_v = json.load(f)
with open('OneForAll/infer_json_pedestrian.json', 'r') as f:
    result_p = json.load(f)

with open('OneForAll/infer.json', 'w') as f:
    result = dict()
    result['results'] = result_v['results'] + result_p['results']
    json.dump(result, f, indent=4)
