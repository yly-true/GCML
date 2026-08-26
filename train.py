import numpy as np
import matplotlib.pyplot as plt

from env_wrapper import FourRoomsDiscrete
from gcml import GCML


def train(env, model, steps=100_000, seed=0):
    rng = np.random.default_rng(seed)
    state_id = rng.integers(env.num_states)

    for _ in range(steps):
        g = env.affordance(state_id)
        feasible = np.flatnonzero(g > 0)

        if len(feasible) == 0:
            state_id = rng.integers(env.num_states)
            continue

        action_id = int(rng.choice(feasible))
        next_id = env.transition(state_id, action_id)

        model.learn_transition(
            env.obs(state_id),
            env.action_one_hot(action_id),
            env.obs(next_id),
            g,
        )

        state_id = next_id

        # occasional restart for broader exploration
        if rng.random() < 0.01:
            state_id = rng.integers(env.num_states)


def imagined_path(env, model, horizon=100):
    actions, latent_states = model.rollout(
        env.obs(env.start_id),
        env.obs(env.goal_id),
        horizon=horizon,
        goal_state_id=env.goal_id,
        start_state_id=env.start_id,
        action_mask_fn=env.affordance,
        transition_fn=env.transition,
    )

    decoded_ids = [model.decode_state(s) for s in latent_states]
    decoded_xy = [env.id_to_state[i] for i in decoded_ids]

    return actions, decoded_xy


def execute_actions(env, actions):
    state_id = env.start_id
    path = [env.id_to_state[state_id]]

    for a in actions:
        state_id = env.transition(state_id, a)
        path.append(env.id_to_state[state_id])

        if state_id == env.goal_id:
            break

    return state_id == env.goal_id, path


def evaluate_model(env, model):
    """Measure whether Q/V encode transitions and G encodes affordances."""
    errors = []
    affordance_correct = 0
    affordance_total = env.num_states * env.num_actions

    for state_id in range(env.num_states):
        o_t = env.obs(state_id)
        predicted_g = model.G @ model.encode(o_t) >= model.affordance_threshold
        actual_g = env.affordance(state_id).astype(bool)
        affordance_correct += int(np.sum(predicted_g == actual_g))

        for action_id in np.flatnonzero(actual_g):
            next_id = env.transition(state_id, int(action_id))
            errors.append(
                model.transition_error(
                    o_t,
                    env.action_one_hot(int(action_id)),
                    env.obs(next_id),
                )
            )

    return {
        "transition_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "affordance_accuracy": affordance_correct / affordance_total,
        "finite_parameters": all(
            np.all(np.isfinite(parameter))
            for parameter in (model.Q, model.V, model.W, model.G)
        ),
    }


def plot(env, imagined_xy, real_xy):
    fig, ax = plt.subplots(figsize=(7, 7))

    # draw map
    free = set(env.free_cells)
    for y in range(env.height):
        for x in range(env.width):
            if (x, y) not in free:
                ax.add_patch(
                    plt.Rectangle((x - 0.5, y - 0.5), 1, 1, alpha=0.35)
                )

    if imagined_xy:
        p = np.asarray(imagined_xy)
        ax.plot(p[:, 0], p[:, 1], "--", marker=".", label="imagined")

    if real_xy:
        p = np.asarray(real_xy)
        ax.plot(p[:, 0], p[:, 1], marker="o", markersize=3, label="executed")

    ax.scatter(*env.start_pos, s=100, label="start")
    ax.scatter(*env.goal_pos, s=140, marker="*", label="goal")

    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(env.height - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("GCML on MiniGrid FourRooms")
    plt.tight_layout()
    plt.savefig("fourrooms_gcml.png", dpi=180)
    plt.show()


def main():
    env = FourRoomsDiscrete(seed=0)

    model = GCML(
        obs_dim=env.num_states,
        action_dim=env.num_actions,
        latent_dim=64,
        noise_std=0.10,
        seed=0,
    )

    print("states:", env.num_states)
    print("start :", env.start_pos)
    print("goal  :", env.goal_pos)

    train(env, model, steps=100_000)

    metrics = evaluate_model(env, model)
    print("transition RMSE:", f"{metrics['transition_rmse']:.4f}")
    print("learned G accuracy (diagnostic only):", f"{metrics['affordance_accuracy']:.2%}")
    print("planning affordance: official FourRooms wall mask")
    print("finite parameters:", metrics["finite_parameters"])

    actions, imagined_xy = imagined_path(env, model, horizon=100)
    success, real_xy = execute_actions(env, actions)

    print("imagined action count:", len(actions))
    print("execution success:", success)

    plot(env, imagined_xy, real_xy)


if __name__ == "__main__":
    main()
