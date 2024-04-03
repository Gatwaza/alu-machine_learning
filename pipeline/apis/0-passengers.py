#!/usr/bin/env python3

"""Retrieves a list of starships """

import requests


def availableShips(passengerCount):
    """
    Retrieves a list of starships from the Star Wars API (Swapi) that can hold
    a given number of passengers or more.

    Args:
        passengerCount (int): The minimum number of passengers a starship should be able to hold.

    Returns:
        list: A list of starship names that can hold at least the specified number of passengers.

    """
    url = 'https://swapi.dev/api/starships/'
    ships = []

    while url:
        response = requests.get(url)
        data = response.json()

        for ship in data['results']:
            if ship['passengers'].isdigit() and int(ship['passengers']) >= passengerCount:
                ships.append(ship['name'])

        url = data['next']

    return ships


if __name__ == "__main__":
    passengerCount = int(input("Enter the number of passengers: "))
    print(availableShips(passengerCount))
