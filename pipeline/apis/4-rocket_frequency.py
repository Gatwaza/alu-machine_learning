#!/usr/bin/env python3
"""
Script to fetch and display the number of launches per rocket
using the unofficial SpaceX API.

Each line will contain the rocket name and the number of launches
separated by a colon and a space, sorted first by the number of
launches in descending order and then alphabetically by rocket
name if the counts are the same.

Example output:
Falcon 9: 195
Falcon 1: 5
Falcon Heavy: 5
"""

import requests

def fetch_rocket_launch_counts():
    """
    Fetches launch data from SpaceX API and counts the number of launches
    per rocket.

    Returns:
        dict: A dictionary with rocket IDs as keys and launch counts as values.
    """
    url = "https://api.spacexdata.com/v4/launches"
    response = requests.get(url)
    response.raise_for_status()  # Raises an HTTPError for bad responses
    
    rocket_dict = {}
    for launch in response.json():
        rocket_id = launch["rocket"]
        if rocket_id in rocket_dict:
            rocket_dict[rocket_id] += 1
        else:
            rocket_dict[rocket_id] = 1
    return rocket_dict

def fetch_rocket_name(rocket_id):
    """
    Fetches the rocket name for a given rocket ID from SpaceX API.

    Args:
        rocket_id (str): The unique ID of the rocket.

    Returns:
        str: The name of the rocket.
    """
    url = f"https://api.spacexdata.com/v4/rockets/{rocket_id}"
    response = requests.get(url)
    response.raise_for_status()  # Raises an HTTPError for bad responses
    
    return response.json()["name"]

def main():
    """
    Main function to execute the script:
    - Fetches and counts rocket launches.
    - Retrieves rocket names.
    - Sorts the results by launch count and rocket name.
    - Prints the results in the specified format.
    """
    rocket_dict = fetch_rocket_launch_counts()

    # List to hold (rocket_name, count) tuples
    rocket_info = []
    
    # Fetch rocket names and prepare for sorting
    for rocket_id, count in rocket_dict.items():
        rocket_name = fetch_rocket_name(rocket_id)
        rocket_info.append((rocket_name, count))

    # Sort by the number of launches (descending) and then by rocket name (alphabetically)
    rocket_info.sort(key=lambda x: (-x[1], x[0]))

    # Print the sorted rockets with their launch counts
    for rocket_name, count in rocket_info:
        print(f"{rocket_name}: {count}")

if __name__ == '__main__':
    main()
