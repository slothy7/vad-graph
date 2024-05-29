import numpy as np


def agc_initialize(
    nominal_gain,
    agc_threshold,
    attack_time,
    release_time,
    sample_rate,
    max_gain_reduction=0.1,
):
    attack_coeff = 1.0 - np.exp(-1.0 / (attack_time * sample_rate))
    release_coeff = 1.0 - np.exp(-1.0 / (release_time * sample_rate))
    min_gain = nominal_gain * max_gain_reduction
    return {
        "nominal_gain": nominal_gain,
        "agc_threshold": agc_threshold,
        "attack_coeff": attack_coeff,
        "release_coeff": release_coeff,
        "gain": nominal_gain,
        "min_gain": min_gain,
    }


def agc_process(samples, agc_state):
    output = np.zeros_like(samples)
    for i, sample in enumerate(samples):
        abs_sample = np.abs(sample * agc_state["gain"])
        if abs_sample > agc_state["agc_threshold"]:
            agc_state["gain"] -= agc_state["attack_coeff"] * (
                agc_state["gain"] - (agc_state["agc_threshold"] / abs_sample)
            )
            agc_state["gain"] = max(agc_state["gain"], agc_state["min_gain"])
        else:
            agc_state["gain"] += agc_state["release_coeff"] * (
                agc_state["nominal_gain"] - agc_state["gain"]
            )

        output_sample = sample * agc_state["gain"]
        output[i] = np.clip(output_sample, -1.0, 1.0)
    return output, agc_state
