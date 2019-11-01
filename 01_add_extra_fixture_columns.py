# add some new metrics to the fixture dataset

from fpl import fpl
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


# region Read Data
"""
READ DATA
"""

fixtures = fpl.read_fixtures()
teams = fpl.read_teams()


# endregion


# region Add team fixture number
"""
ADD TEAM FIXTURE NUMBER
"""

fixtures["team_h_game_number"] = \
    pd.Series(np.zeros(len(fixtures), dtype="int"))

fixtures["team_a_game_number"] = \
    pd.Series(np.zeros(len(fixtures), dtype="int"))

game_number = np.arange(1, 39)

team_id = np.arange(0, 20)
for i in team_id:
    team_h_bool = fixtures["team_h"] == i
    team_a_bool = fixtures["team_a"] == i
    team_bool = np.logical_or(team_h_bool, team_a_bool)

    team_df = fixtures.loc[team_bool, :]

    team_df = team_df.sort_values(by=["kickoff_time"]).reset_index()

    team_df["game_number"] = game_number

    team_df = team_df.set_index("index")

    fixtures.loc[team_h_bool, "team_h_game_number"] = team_df.loc[
        team_h_bool, "game_number"]

    fixtures.loc[team_a_bool, "team_a_game_number"] = team_df.loc[
        team_a_bool, "game_number"]

# endregion


# region Team Points
"""
TEAM POINTS
"""

team_h_points = np.zeros(len(fixtures), dtype="int")
team_a_points = np.zeros(len(fixtures), dtype="int")

for i in fixtures.index:

    if fixtures.loc[i, "team_h_score"] > fixtures.loc[i, "team_a_score"]:
        team_h_points[i] = 3
        team_a_points[i] = 0
    elif fixtures.loc[i, "team_a_score"] > fixtures.loc[i, "team_h_score"]:
        team_h_points[i] = 0
        team_a_points[i] = 3
    else:
        team_h_points[i] = 1
        team_a_points[i] = 1

fixtures["team_h_points"] = pd.Series(team_h_points)
fixtures["team_a_points"] = pd.Series(team_a_points)

# endregion


# region Team Points at n Games
"""
TEAM POINTS AFTER N GAMES
"""

team_points = np.zeros((38, 20), dtype="int")

fixtures["team_h_running_points"] = \
    pd.Series(np.zeros(len(fixtures), dtype="int"))

fixtures["team_a_running_points"] = \
    pd.Series(np.zeros(len(fixtures), dtype="int"))

for i in team_id:

    team_h_bool = fixtures["team_h"] == i
    team_a_bool = fixtures["team_a"] == i
    team_bool = np.logical_or(team_h_bool, team_a_bool)

    team_df = fixtures.loc[team_bool, :]

    team_df = team_df.sort_values(by=["kickoff_time"]).reset_index()

    team_df["team_points"] = pd.Series(np.zeros(len(team_df), dtype="int"))

    for j in team_df.index:
        if team_df.loc[j, "team_h"] == i:
            team_df.loc[j, "team_points"] = team_df.loc[j, "team_h_points"]
        else:
            team_df.loc[j, "team_points"] = team_df.loc[j, "team_a_points"]

    team_df["running_points"] = np.cumsum(team_df["team_points"])

    team_points[:, i] = team_df["running_points"].values

    team_df = team_df.set_index("index")

    fixtures.loc[team_h_bool, "team_h_running_points"] = team_df.loc[
        team_h_bool, "running_points"]

    fixtures.loc[team_a_bool, "team_a_running_points"] = team_df.loc[
        team_a_bool, "running_points"]

print(fixtures.head(38))

team_points = np.append(np.zeros((1, 20), dtype="int"), team_points, axis=0)
team_points = pd.DataFrame(team_points, index=np.arange(0, 39),
                           columns=teams["short_name"].values)

print(team_points.head())

# endregion


# region Save Data

fixtures.to_csv(r"data/csv/fixtures.csv", index=False, index_label=False)
team_points.to_csv(r"data/csv/team_points.csv", index=False, index_label=False)

# endregion
