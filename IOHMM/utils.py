"""
Those are model utilities.

Could be integrated as IOHMM classes methods,
but may not be generic enough (1 sequence, univariate emissions hypothesis)

Thus, this module is expected to import from IOHMM, not the other way around.

May be renamed
"""

import numpy as np
import pandas as pd

from IOHMM import UnSupervisedIOHMM

SEED = 1


def get_posterior_proba(
    model: UnSupervisedIOHMM, data: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Get hidden states posterior probabilities from a data sequence and fitted HMM model

    If data is not provided that means it is assumed already set into model
    """
    assert model.num_seqs == 1, "Assumes 1 sequence"

    # Set data if provided
    if data is not None:
        model.set_data([data])

    # 1 Expectation iteration from the EM (Baum-Welch) training algorithm allows you to
    # make a forward-backward pass and get back missing attributes from training
    model.E_step()

    # Extract posterior proba
    post_prob = pd.DataFrame(
        np.exp(model.log_gammas[0]),  # 1 array by sequence/df
        columns=[f"state {s}" for s in range(model.num_states)],
        index=model.dfs_logStates[0][0].index,
    )

    return post_prob


def get_fitted_hidden_states_from_post_proba(post_prob: pd.DataFrame) -> pd.Series:
    """
    Compute hidden states maximizing states posterior probabilities
    """
    return post_prob.T.reset_index(drop=True).idxmax().rename("hidden_states")


def get_fitted_hidden_states(
    model: UnSupervisedIOHMM, data: pd.DataFrame = None
) -> pd.Series:
    """
    Wrapping posterior probabilities computation + states maximizing probabilities derivation
    """
    post_prob = get_posterior_proba(model=model, data=data)
    fitted_hidden_states = get_fitted_hidden_states_from_post_proba(post_prob)
    return fitted_hidden_states


def get_fitted_values(
    model: UnSupervisedIOHMM, hidden_states: pd.Series, data: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Get fitted values from a data sequence and fitted HMM model
    """
    assert all(
        len(model.responses_emissions[emis]) == 1 for emis in range(model.num_emissions)
    ), "expecting univariate emissions output"
    assert model.num_seqs == 1, "Assumes 1 sequence"

    # Set data if provided
    if data is not None:
        assert data.index.equals(hidden_states.index), (
            "data and hidden_states must share index"
        )
        model.set_data([data])

    # Collecting fitted values from emission models
    res = {}
    for emis in range(model.num_emissions):
        X = pd.DataFrame(
            model.inp_emissions_all_sequences[emis],
            index=model.dfs_logStates[0][0].index,
        )  # assumes 1 sequence
        res[model.responses_emissions[emis][0]] = pd.Series(index=X.index)
        for st in range(model.num_states):
            is_sample_from_st = hidden_states == st
            res[model.responses_emissions[emis][0]].loc[is_sample_from_st] = (
                model.model_emissions[st][emis].predict(X[is_sample_from_st].values)[
                    :, 0
                ]  # assumes univariate emissions
            )

    return pd.DataFrame(res)


def hidden_state_period_span(st: int, hidden_states: pd.Series):
    """
    Extract 1-state period spans from a sequence of hidden states indexed with time.
    Return a list of lists of time spans bounds
    """
    st_dates = hidden_states[hidden_states == st].index
    time_gap_to_next_st_timestamp = st_dates.to_series().diff().dt.seconds / 3600
    i = 0
    intervals = []
    while i < len(time_gap_to_next_st_timestamp):
        di = time_gap_to_next_st_timestamp.index[i]
        if i == (len(time_gap_to_next_st_timestamp) - 1):
            intervals.append([di, di])
            return intervals
        while ((i + 1) < len(time_gap_to_next_st_timestamp)) and (
            time_gap_to_next_st_timestamp.iloc[i + 1] == 1
        ):
            i += 1
        df = time_gap_to_next_st_timestamp.index[i]
        intervals.append([di, df])
        i += 1
    return intervals


def get_emissions_coef_df(model: UnSupervisedIOHMM) -> pd.DataFrame:
    """
    Format all HMM emissions models coefs into 1 dataframe
    """
    assert all(
        len(model.responses_emissions[emis]) == 1 for emis in range(model.num_emissions)
    ), "expecting univariate emissions output"

    coef_df = []
    for st in range(model.num_states):
        for emis in range(model.num_emissions):
            fit_intercept = model.model_emissions[st][
                emis
            ].fit_intercept  # doesn't change with state
            df = pd.DataFrame(
                data=model.model_emissions[st][emis].coef[
                    0, :
                ],  # assuming univariate output
                index=(["Intercept"] if fit_intercept else [])
                + model.covariates_emissions[emis],
                columns=["coef_value"],
            )
            df.index.name = "coef_name"
            df["state"] = st
            df["target"] = f"target_{emis}"
            coef_df.append(df)

    return pd.concat(coef_df, ignore_index=False).reset_index()


def simulate_states_from_exogenous_input(
    model: UnSupervisedIOHMM,
    state_ini: int,
    data: pd.DataFrame = None,
    seed: int = SEED,
) -> pd.Series:
    """
    Simulates hidden states from a model with set (or not yet) data.

    States simulation is based on exogenous information only.
    In other words, it assumes next states inference to NOT rely on past process values,
    thus there is no not need for inter-linked emissions models simulation.
    """
    assert model.num_seqs == 1, "Assumes 1 sequence"
    if seed is not None:
        np.random.seed(seed)

    # Set data if provided
    if data is not None:
        model.set_data([data])

    # Computing transition probabilities at once, for all (t, state_t_1, state_t)
    transition_proba = model.predict_transition_proba()[0]  # assumes 1 sequence

    # Iterative state simulation
    states = [state_ini]
    for t in range(transition_proba.shape[0]):
        transition_proba_to_state_t = np.squeeze(transition_proba[t, states[-1], :])
        states.append(
            np.random.choice(range(model.num_states), p=transition_proba_to_state_t)
        )

    return pd.Series(
        states, index=model.dfs_logStates[0][0].index, name="simulated_states"
    )  # assumes 1 sequence
