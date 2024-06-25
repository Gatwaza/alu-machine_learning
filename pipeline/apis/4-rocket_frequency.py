#!/usr/bin/env python3
"""Pipeline Api"""
import requests

if __name__ == '__main__':
    """pipeline api"""
    url = "https://api.spacexdata.com/v4/launches"
    r = requests.get(url)
    rocket_dict = {}

    # Count the number of launches per rocket ID
    for launch in r.json():
        rocket_id = launch["rocket"]
        if rocket_id in rocket_dict:
            rocket_dict[rocket_id] += 1
        else:
            rocket_dict[rocket_id] = 1

    rocket_info = []
    
    # Fetch rocket names and prepare them for sorting
    for rocket_id, count in rocket_dict.items():
        rurl = "https://api.spacexdata.com/v4/rockets/" + rocket_id
        req = requests.get(rurl)
        rocket_name = req.json()["name"]
        rocket_info.append((rocket_name, count))

    # Sort by the number of launches (descending) and then by rocket name (alphabetically)
    rocket_info.sort(key=lambda x: (-x[1], x[0]))

    # Print the sorted rockets with their launch counts
    for rocket_name, count in rocket_info:
        print(f"{rocket_name}: {count}")
