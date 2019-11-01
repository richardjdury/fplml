import pandas as pd
import numpy as np


# region Read Data


def read_fixtures():
    return pd.read_csv("data/csv/fixtures.csv")


def read_teams():
    return pd.read_csv("data/csv/teams.csv")


def read_players():
    return pd.read_csv("data/csv/players.csv")


def read_pp():
    return pd.read_csv("data/csv/pp.csv")


# endregion

# region Add Team

def add_team(pp, fixture, was_home, onehot=False):
    fixtures = read_fixtures()

    teams = read_teams()

    team = np.zeros(len(pp), dtype="int")

    for i in pp.index:

        if was_home[i]:
            team[i] = fixtures.loc[fixture[i], "team_h"]
        else:
            team[i] = fixtures.loc[fixture[i], "team_a"]

    if onehot:
        from sklearn.preprocessing import OneHotEncoder
        ohe = OneHotEncoder(sparse=False, categories="auto")
        ohe_team = ohe.fit_transform(team.reshape(-1, 1))
        team_cols = ["team_" + s for s in teams["short_name"]]
        pp[team_cols] = pd.DataFrame(ohe_team == 1)
    else:
        pp["team"] = pd.Series(team)

    return pp


# endregion

# region Add Opponent Team

def add_opponent_team(pp, fixture, was_home, onehot=False):
    fixtures = read_fixtures()

    teams = read_teams()

    opponent_team = np.zeros(len(pp), dtype="int")

    for i in pp.index:

        if was_home[i]:
            opponent_team[i] = fixtures.loc[fixture[i], "team_a"]
        else:
            opponent_team[i] = fixtures.loc[fixture[i], "team_h"]

    if onehot:
        from sklearn.preprocessing import OneHotEncoder
        ohe = OneHotEncoder(sparse=False, categories="auto")
        ohe_opponent_team = ohe.fit_transform(opponent_team.reshape(-1, 1))
        opponent_team_cols = ["opponent_team_" +
                              s for s in teams["short_name"]]
        pp[opponent_team_cols] = pd.DataFrame(ohe_opponent_team == 1)
    else:
        pp["opponent_team"] = pd.Series(opponent_team)

    return pp


# endregion

# region Add Player Position
def add_player_position(pp, player_id, onehot=False):
    players = read_players()

    position = players.loc[np.array(player_id), "position"].values

    if onehot:
        from sklearn.preprocessing import OneHotEncoder
        ohe = OneHotEncoder(sparse=False, categories="auto")
        ohe_position = ohe.fit_transform(position.reshape(-1, 1))
        pp[["GKP", "DEF", "MID", "FWD"]] = pd.DataFrame(ohe_position == 1)
    else:
        pp["position"] = pd.Series(position)

    return pp


# endregion

# region Calculate Form
def calculate_form(pp, player_id, column, n_games, sigma):
    from scipy.signal import convolve

    unique_player_id = np.unique(player_id)

    response = np.zeros((2 * n_games) + 1, dtype="float")
    response[(n_games + 1):] = norm_pdf(np.array(range(n_games)), 0, sigma)
    response = response / np.sum(response)

    form = np.zeros(len(player_id), dtype="float")

    for i in unique_player_id:
        player_column = column.loc[player_id == i]
        player_values = player_column.values
        player_index = player_column.index
        player_form = convolve(player_values, response, mode="same")
        form[list(player_index)] = player_form

    if sigma == np.inf:
        sigma_str = "inf"
    else:
        sigma_str = str(round(sigma, 1)).replace(".", "p")

    pp[column.name + "_form"] = pd.Series(form)
    return pp


# endregion

# region Player Value

def add_player_value(pp, player_id, player_value):
    unique_player_id = np.unique(player_id)

    prev_value = np.zeros(len(player_id), dtype="float")
    diff_value = np.zeros(len(player_id), dtype="float")

    for i in unique_player_id:
        player_col = player_value[player_id == i]
        player_index = player_col.index
        player_values = player_col.values
        player_values_shifted = shift_float(player_values, 1)
        prev_value[list(player_index)] = player_values_shifted
        prev_value[player_index[0]] = player_values[0]
        diff_value[list(player_index)] = player_values_shifted - player_values[
            0]
        diff_value[player_index[0]] = 0

    pp["prev_value"] = pd.Series(prev_value)
    pp["diff_value"] = pd.Series(diff_value)

    return pp


# endregion

# region Played Previous Game

def played_prev_game(pp, player_id, minutes):
    unique_player_id = np.unique(player_id)

    played_last_game = np.zeros(len(player_id), dtype="bool")

    for i in unique_player_id:
        player_col = minutes.loc[player_id == i]
        player_index = player_col.index
        player_values = player_col.values
        player_values_bool = player_values > 60
        player_values_shifted = shift_bool(player_values_bool, 1)
        played_last_game[list(player_index)] = player_values_shifted

    pp["played_last_game"] = pd.Series(played_last_game)

    return pp


# endregion

# region Add Team's Game Number

def add_team_game_number(pp, fixture_id, was_home):
    fixtures = read_fixtures()

    team_game_number = np.zeros(len(fixture_id), dtype="int")

    for i in fixture_id.index:

        if was_home[i]:
            team_game_number[i] = fixtures.loc[fixture_id[i],
                                               "team_h_game_number"]
        else:
            team_game_number[i] = fixtures.loc[fixture_id[i],
                                               "team_a_game_number"]

    pp["team_game_number"] = pd.Series(team_game_number)
    return pp


# endregion

# region Add Player Points Contribution

def add_player_points_contribution(pp, team_id, team_game_number,
                                   total_points):
    player_points_contribution = np.zeros(len(team_id), dtype="float")
    team_total_points = np.zeros(len(team_id), dtype="int")

    for i in team_id.index:
        team_fixture_bool = np.logical_and(team_id == team_id[i],
                                           team_game_number ==
                                           team_game_number[i])

        total_team_points = np.sum(total_points[team_fixture_bool])

        team_total_points[i] = total_team_points
        player_points_contribution[i] = total_points[i] / total_team_points

    pp["player_points_contribution"] = pd.Series(player_points_contribution)
    pp["team_total_points"] = pd.Series(team_total_points)
    return pp


# endregion

# region Calculate Team League Form

def add_team_league_points(pp, fixture_id, was_home):
    fixtures = read_fixtures()

    team_points = np.zeros(len(pp), dtype="int")

    for i in pp.index:

        if was_home[i]:
            team_points[i] = fixtures.loc[fixture_id[i], "team_h_points"]
        else:
            team_points[i] = fixtures.loc[fixture_id[i], "team_a_points"]

    pp["team_league_points"] = pd.Series(team_points)
    return pp


# endregion

# region Opponent Team Form

def add_opponent_team_league_form(pp, fixture_id, was_home, team_league_form):

    opponent_team_form = np.zeros(len(pp), dtype="float")

    for i in pp.index:

        current_fixture = fixture_id[i]
        opponent_was_home = not was_home[i]

        opponent_bool = np.logical_and(fixture_id == current_fixture,
                                       was_home == opponent_was_home)

        tmp_team_league_form = team_league_form[opponent_bool]
        opponent_team_form[i] = tmp_team_league_form.values[0]

    pp["opponent_team_league_points_form"] = pd.Series(opponent_team_form)
    return pp


# endregion

def add_opponent_team_points_form(pp, fixture_id, was_home, team_points_form):

    opponent_team_points_form = np.zeros(len(pp), dtype="float")

    for i in pp.index:

        current_fixture = fixture_id[i]
        opponent_was_home = not was_home[i]

        opponent_bool = np.logical_and(fixture_id == current_fixture,
                                       was_home == opponent_was_home)

        tmp_team_points_form = team_points_form[opponent_bool]
        opponent_team_points_form[i] = tmp_team_points_form.values[0]

    pp["opponent_team_total_points_form"] = \
        pd.Series(opponent_team_points_form)
    return pp


# endregion


"""
EXTRA FUNCTIONS
"""

# region Normal Distribution

import math

def norm_pdf(x, mu, sigma):
    if np.isinf(sigma):
        return np.ones(len(x), dtype="float") * (1 / len(x))
    else:
        return (1 / np.sqrt(2 * math.pi * (sigma ** 2))) * \
               np.exp(-((x - mu) ** 2) / (2 * (sigma ** 2)))


# endregion

# region ShiftArrays

def shift_float(xs, n):
    if n >= 0:
        return np.concatenate((np.full(n, xs[0]), xs[:-n]))
    else:
        return np.concatenate((xs[-n:], np.full(-n, xs[-1])))


def shift_bool(xs, n):
    if n >= 0:
        return np.concatenate((np.full(n, xs[0]), xs[:-n]))
    else:
        return np.concatenate((xs[-n:], np.full(-n, xs[-1])))


# endregion

# region Balanced Subsampling
def balancedSubsampling(X, Y, random_state=42):
    # label information
    Y_unique = np.unique(Y)
    Y_n = len(Y_unique)

    # Count label population
    count = np.zeros(Y_n, dtype="int")
    for i in range(Y_n):
        count[i] = np.sum(Y == Y_unique[i])

    min_samples = np.min(count)
    balanced_X = pd.DataFrame()
    balanced_Y = pd.DataFrame()
    remaining_X = pd.DataFrame()
    remaining_Y = pd.DataFrame()

    for i in range(Y_n):
        # calculate sample indicies
        tmp_X = X.loc[Y == Y_unique[i], :]
        np.random.seed(10)
        index_sample = np.random.choice(list(tmp_X.index), min_samples,
                                        replace=False)
        # remaining index
        rem_index_sample = list(set(tmp_X.index) - set(index_sample))

        # appened dataframes
        balanced_X = balanced_X.append(tmp_X.loc[index_sample, :])
        balanced_Y = pd.concat([balanced_Y, Y.loc[index_sample]])

        # remaining_X
        remaining_X = remaining_X.append(tmp_X.loc[rem_index_sample, :])
        remaining_Y = pd.concat([remaining_Y, Y.loc[rem_index_sample]])

    return balanced_X.sort_index(), balanced_Y.sort_index(), remaining_X.sort_index(), remaining_Y.sort_index()

# endregion
