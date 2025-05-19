import numpy as np
import matplotlib.pyplot as plt
import json

WAYPOINT_JSON = "./all_waypoints.json"

with open(WAYPOINT_JSON, 'r', encoding='utf-8') as f:
    wp_json = json.load(f)

points = []
for wp_dict in wp_json:
    for delta in wp_dict['deltas']:
        points.append(delta)

points_array = np.array(points)

x = points_array[:, 0]
y = points_array[:, 1]

x_min, x_max = np.min(x), np.max(x)
y_min, y_max = np.min(y), np.max(y)

print(f"x range: [{x_min}, {x_max}]")
print(f"y range: [{y_min}, {y_max}]")

plt.figure(figsize=(8, 8))
plt.scatter(y, x, s=1, alpha=0.5)

plt.gca().set_aspect('equal')
plt.xlabel("Y (right)")
plt.ylabel("X (forward)")
plt.title("Distribution of Relative Positions every 0.5s")

plt.savefig("./waypoint_deltas.png", dpi=300, bbox_inches='tight')
plt.close() 
