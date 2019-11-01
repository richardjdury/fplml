# Read in json data, clean and save as csv in data/csv

import collections
import json
import numpy as np
import pandas as pd

# region Script Settings
"""
SCRIPT SETTINGS
"""

pandas_display_width = 150
pd.set_option("display.width", pandas_display_width)
pd.set_option("display.max_columns", None)

# endregion

# region Teams
"""
TEAMS
"""

teams_filename = "data/json/teams/teams.json"

with open(teams_filename, encoding="utf8") as f:
    teams = pd.DataFrame(json.load(f))

teams_columns_to_drop = ["current_event_fixture", "next_event_fixture",
                         "code", "unavailable", "position", "played", "win",
                         "loss", "draw", "points", "form", "link_url",
                         "team_division"]

teams = teams.drop(columns=teams_columns_to_drop)

teams_restart_column_index = ["id"]
teams[teams_restart_column_index] = teams[teams_restart_column_index] - 1

print("team row index equals id: " +
      str(np.array_equal(teams.index, teams["id"].values)))

teams.to_csv(r"data/csv/teams.csv", index=False, index_label=False)

# endregion

# region Fixtures
"""
FIXTURES
"""

fixtures_filename = "data/json/fixtures/fixtures.json"

with open(fixtures_filename, encoding="utf8") as f:
    fixtures = pd.DataFrame(json.load(f))

fixtures_columns_to_drop = ["kickoff_time_formatted", "started",
                            "event_day", "deadline_time",
                            "deadline_time_formatted", "stats", "code",
                            "finished", "minutes", "provisional_start_time",
                            "finished_provisional"]

fixtures = fixtures.drop(columns=fixtures_columns_to_drop)

fixtures_restart_column_index = ["id", "team_a", "team_h", "event"]
fixtures[fixtures_restart_column_index] = \
    fixtures[fixtures_restart_column_index] - 1

fixtures = fixtures.sort_values(by=["id"]).reset_index(drop=True)

print("fixtures row index equals id: " +
      str(np.array_equal(fixtures.index, fixtures["id"].values)))

fixtures.to_csv(r"data/csv/fixtures.csv", index=False, index_label=False)

# endregion

# region Players
"""
PLAYERS
"""

players_filename = "data/json/elements/elements.json"

with open(players_filename, encoding="utf8") as f:
    players = pd.DataFrame(json.load(f))

player_ids = np.unique(players["id"])

n_players = len(player_ids)

players_columns_to_keep = ["id", "first_name", "second_name", "web_name",
                           "element_type", "team", "squad_number"]
players = players[players_columns_to_keep]

players = players.rename(columns={"element_type": "position"})

players = players.sort_values(by=["id"]).reset_index(drop=True)

players_restart_column_index = ["id", "team", "position"]
players[players_restart_column_index] = \
    players[players_restart_column_index] - 1

players["squad_number"] = players["squad_number"].fillna(0).astype("int")

print("players row index equals id: " +
      str(np.array_equal(players.index, players["id"].values)))

players.to_csv(r"data/csv/players.csv", index=False, index_label=False)

# endregion

# region Player Performance
"""
PLAYER PERFORMANCE
"""

pp_od = collections.OrderedDict()

found_column_names = False
loop_counter = 0

for i in player_ids:

    with open("data/json/element-summary/" + str(i) + ".json") as f:
        tmp_pp = json.load(f)

    if not found_column_names:
        cn = tmp_pp["history"][0].keys()
        found_column_names = True

    n_fixtures = len(tmp_pp["history"])

    for j in range(0, n_fixtures):
        pp_od[loop_counter] = tmp_pp["history"][j]
        loop_counter = loop_counter + 1

pp = pd.DataFrame.from_dict(pp_od, orient="index")

pp_columns_to_drop = ["id", "kickoff_time_formatted"]
pp = pp.drop(columns=pp_columns_to_drop)

pp = pp.rename(columns={"element": "id"})

pp = pp.sort_values(by=["id", "kickoff_time"]).reset_index(drop=True)

pp_restart_column_index = ["id", "fixture", "opponent_team"]
pp[pp_restart_column_index] = pp[pp_restart_column_index] - 1

pp.to_csv(r"data/csv/pp.csv", index=False, index_label=False)

# endregion

print("finished")
