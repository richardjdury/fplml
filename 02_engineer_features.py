# add some new metrics to the pp dataset

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

np.seterr(divide='ignore', invalid='ignore')

# endregion

# region Read Data

fixtures = fpl.read_fixtures()
teams = fpl.read_teams()
players = fpl.read_players()
pp = fpl.read_pp()

pp_prepared = pd.DataFrame()

# endregion

# region Categorical labels

label_values = [1, 2, 3, 4]
label_names = ["Poor", "Average", "Good", "Excellent"]

labels_cat = pd.cut(
    pp["total_points"],
    bins=[-np.inf, 0.5, 3.5, 7.5, np.inf],
    labels=label_values)

# endregion

"""
BEGIN ADDING FEATURES
"""

# region Add "was_home" feature

pp_prepared["was_home"] = pp["was_home"]

# endregion

# region Add Points Per Minute

pp["total_points_per_minute"] = (pp["total_points"] / pp["minutes"]).fillna(0)

# endregion

# region Add player's team

pp = fpl.add_team(pp, pp["fixture"], pp["was_home"], False)
pp_prepared = fpl.add_team(pp_prepared, pp["fixture"], pp["was_home"], True)

# endregion

# region Top 6

pp_prepared["team_top_6"] = pd.Series(
    np.in1d(pp["team"].values, np.array([0, 5, 11, 12, 13, 16])))

pp_prepared["opponent_team_top_6"] = pd.Series(
    np.in1d(pp["opponent_team"].values, np.array([0, 5, 11, 12, 13, 16])))

# endregion

# region Add player's opponent team

pp_prepared = fpl.add_opponent_team(pp_prepared, pp["fixture"],
                                    pp["was_home"], True)

# endregion

# region Add Player's position

pp_prepared = fpl.add_player_position(pp_prepared, pp["id"], True)

# endregion

# region Add Player Values

pp_prepared = fpl.add_player_value(pp_prepared, pp["id"], pp["value"])

# endregion

# region Played Previous Game

pp_prepared = fpl.played_prev_game(pp_prepared, pp["id"], pp["minutes"])

# endregion

# region Player Points Contribution

pp = fpl.add_team_game_number(pp, pp["fixture"], pp["was_home"])
pp = fpl.add_player_points_contribution(pp, pp["team"],
                                        pp["team_game_number"],
                                        pp["total_points"])

# endregion

# region Add Team League Points

pp = fpl.add_team_league_points(pp, pp["fixture"], pp["was_home"])

# endregion

# region Add Category Frequency

pp_prepared = fpl.previous_label_frequency(pp_prepared, labels_cat, pp["id"], label_names)

# endregion

# region Calculate Form

n_games = 2
sigma = np.inf

pp_prepared = fpl.calculate_form(pp_prepared, pp["id"], pp["minutes"],
                                 n_games, sigma)

pp_prepared = fpl.calculate_form(pp_prepared, pp["id"], pp["total_points"],
                                 n_games, sigma)

pp_prepared = fpl.calculate_form(pp_prepared, pp["id"], pp["threat"],
                                 n_games, sigma)

pp_prepared = fpl.calculate_form(pp_prepared, pp["id"], pp["bps"],
                                 3, 1)

pp_prepared = fpl.calculate_form(pp_prepared, pp["id"],
                                 pp["total_points_per_minute"], n_games,
                                 sigma)

pp_prepared = fpl.calculate_form(pp_prepared, pp["id"],
                                 pp["selected"], n_games, sigma)

pp_prepared = fpl.calculate_form(pp_prepared, pp["id"],
                                 pp["transfers_balance"], n_games, sigma)

pp_prepared = fpl.calculate_form(pp_prepared, pp["id"],
                                 pp["player_points_contribution"], n_games,
                                 sigma)

pp_prepared = fpl.calculate_form(pp_prepared, pp["id"],
                                 pp["team_total_points"], n_games, sigma)

pp_prepared = fpl.calculate_form(pp_prepared, pp["id"],
                                 pp["team_league_points"],
                                 n_games, sigma)

# region Exprected Points

pp_prepared["expected_points"] = pp_prepared["minutes_form"]*pp_prepared[
    "total_points_per_minute_form"]

pp_prepared = pp_prepared.drop(columns=["total_points_per_minute_form",
                                        "total_points_form"])

# endregion

# endregion

# region Add Opponent Team League Form

pp_prepared = fpl.add_opponent_team_league_form(pp_prepared, pp["fixture"],
                                                pp["was_home"], pp_prepared[
                                                   "team_league_points_form"])

pp_prepared["team_form_difference"] = pp_prepared["team_league_points_form"]\
                                      - pp_prepared[
                                          "opponent_team_league_points_form"]

# endregion

# region Add Opponent Team Points Form

pp_prepared = fpl.add_opponent_team_points_form(pp_prepared, pp["fixture"],
                                                pp["was_home"], pp_prepared[
                                                    "team_total_points_form"])

pp_prepared["team_total_points_difference"] = pp_prepared[
    "team_total_points_form"]-pp_prepared["opponent_team_total_points_form"]

# endregion

print(pp_prepared.head())

# region Save Data

"""
SAVE DATA
"""

pp_prepared.to_csv(r"data/csv/pp_prepared.csv", index=False, index_label=False)

pp["total_points"].to_csv(r"data/csv/label_values.csv", index=False,
                          index_label=False, header="label")

labels_cat.to_csv(r"data/csv/label_cat.csv", index=False, index_label=False,
                  header="label")

# endregion
