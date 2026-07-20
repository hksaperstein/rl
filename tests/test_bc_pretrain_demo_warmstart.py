"""Sim-independent unit tests for scripts/bc_pretrain_demo_warmstart.py's
replay-then-regress driver logic - no Isaac Lab import needed (mirrors
tests/test_distillation_data_collection.py's own scope split: pure
math/mechanics exercised here in isolation from a live simulated env, via
stub trajectories/a stub env). Run via:

/home/saps/IsaacLab/_isaac_sim/python.sh -m pytest tests/test_bc_pretrain_demo_warmstart.py -v -p no:launch_testing

(plain python3/pytest lacks torch/rsl_rl in this environment - see
project_pytest-needs-isaac-sim-python memory and
tasks/franka/distillation.py's own module docstring.)

Task 2 of docs/superpowers/plans/2026-07-19-d8-d10-demo-warmstart-
implementation.md.
"""

from __future__ import annotations

import pytest
import torch

import argparse

from scripts.bc_pretrain_demo_warmstart import (
    bc_pretrain_shape,
    bc_pretrain_until_plateau,
    build_replay_action_fn,
    demo_trajectory_to_raw_actions,
    get_default_arm_joint_pos,
    main,
    pool_trajectory_batches,
    replay_trajectory_to_paired_batch,
    required_episode_length_s,
)
from tasks.franka.demo_action_mapping import gripper_target_to_raw_action, joint_pos_to_raw_action

OBS_DIM = 10
NUM_ACTIONS = 8  # 7 arm + 1 gripper, matching the real schema


class _StubData:
    pass


class _StubRobot:
    """Duck-types isaaclab.assets.Articulation's own
    find_joints/data.default_joint_pos contract, exactly what
    get_default_arm_joint_pos needs."""

    def __init__(self, num_envs: int, num_joints: int, default_joint_pos: torch.Tensor | None = None):
        self.data = _StubData()
        if default_joint_pos is None:
            default_joint_pos = torch.zeros(num_envs, num_joints)
        self.data.default_joint_pos = default_joint_pos
        self._num_joints = num_joints

    def find_joints(self, _name_keys):
        # First 7 joints are the "arm" joints in this stub's own fixed layout
        # (mirrors the script's own _StubRobot).
        return list(range(7)), [f"panda_joint{i + 1}" for i in range(7)]


class _StubEnv:
    """Deterministic, physics-free stub satisfying collect_rollout's
    reset()/step() contract PLUS get_default_arm_joint_pos's own
    unwrapped.scene["robot"] contract - mirrors
    tests/test_distillation_data_collection.py's own _StubEnv precedent,
    extended with the extra attributes this module's own functions need."""

    def __init__(self, num_envs: int = 1, obs_dim: int = OBS_DIM, default_joint_pos: torch.Tensor | None = None):
        self.num_envs = num_envs
        self._obs_dim = obs_dim
        self._state = torch.zeros(num_envs, obs_dim)
        self._step_count = 0
        self.reset_calls = 0
        self.step_calls = 0
        self._robot = _StubRobot(num_envs, num_joints=9, default_joint_pos=default_joint_pos)
        self.scene = {"robot": self._robot}

    @property
    def unwrapped(self):
        return self

    def _obs(self):
        return {"policy": torch.full((self.num_envs, self._obs_dim), float(self._step_count))}

    def reset(self):
        self._step_count = 0
        self.reset_calls += 1
        return self._obs()

    def step(self, actions):
        assert actions.shape == (self.num_envs, NUM_ACTIONS)
        self._step_count += 1
        self.step_calls += 1
        reward = torch.zeros(self.num_envs)
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        return self._obs(), reward, done, {}


def _make_trajectory(num_steps: int, seed: int = 0) -> dict:
    gen = torch.Generator().manual_seed(seed)
    arm = torch.randn(num_steps, 7, generator=gen)
    half = num_steps // 2
    gripper = torch.cat([torch.full((half, 2), 0.04), torch.full((num_steps - half, 2), 0.0)], dim=0)
    return {"arm_joint_pos_target": arm, "gripper_target": gripper}


class TestRequiredEpisodeLengthS:
    def test_matches_formula(self):
        assert required_episode_length_s(848, decimation=2, sim_dt=0.01, safety_margin=1.2) == pytest.approx(848 * 2 * 0.01 * 1.2)

    def test_default_safety_margin_adds_headroom(self):
        exact = required_episode_length_s(100, decimation=2, sim_dt=0.01, safety_margin=1.0)
        with_margin = required_episode_length_s(100, decimation=2, sim_dt=0.01)
        assert with_margin > exact

    def test_non_positive_num_steps_raises(self):
        with pytest.raises(ValueError):
            required_episode_length_s(0, decimation=2, sim_dt=0.01)
        with pytest.raises(ValueError):
            required_episode_length_s(-5, decimation=2, sim_dt=0.01)


class TestDemoTrajectoryToRawActions:
    def test_matches_direct_conversion_functions(self):
        trajectory = _make_trajectory(num_steps=6, seed=1)
        default_joint_pos = torch.randn(7)
        raw = demo_trajectory_to_raw_actions(trajectory, default_joint_pos, arm_scale=0.5)
        assert raw.shape == (6, 8)

        expected_arm = joint_pos_to_raw_action(trajectory["arm_joint_pos_target"], default_joint_pos, scale=0.5)
        expected_gripper = gripper_target_to_raw_action(trajectory["gripper_target"])
        assert torch.allclose(raw[:, :7], expected_arm)
        assert torch.allclose(raw[:, 7:8], expected_gripper)

    def test_column_order_is_arm_then_gripper(self):
        """Gripper closed (raw < 0) for the whole trajectory, arm target
        identical to default (raw == 0) - unambiguous per-column check that
        columns 0:7 are the arm's own raw actions and column 7 is the
        gripper's, not swapped."""
        num_steps = 4
        default_joint_pos = torch.zeros(7)
        trajectory = {
            "arm_joint_pos_target": default_joint_pos.unsqueeze(0).expand(num_steps, -1).clone(),
            "gripper_target": torch.zeros(num_steps, 2),  # close_target
        }
        raw = demo_trajectory_to_raw_actions(trajectory, default_joint_pos)
        assert torch.allclose(raw[:, :7], torch.zeros(num_steps, 7))
        assert (raw[:, 7] < 0).all()


class TestBuildReplayActionFn:
    def test_returns_rows_in_order_and_increments(self):
        raw_actions = torch.arange(3 * NUM_ACTIONS, dtype=torch.float32).view(3, NUM_ACTIONS)
        action_fn = build_replay_action_fn(raw_actions)
        obs = torch.zeros(2, OBS_DIM)
        for i in range(3):
            action = action_fn(obs)
            assert action.shape == (2, NUM_ACTIONS)
            assert torch.allclose(action, raw_actions[i].unsqueeze(0).expand(2, -1))

    def test_raises_past_trajectory_length(self):
        raw_actions = torch.zeros(2, NUM_ACTIONS)
        action_fn = build_replay_action_fn(raw_actions)
        obs = torch.zeros(1, OBS_DIM)
        action_fn(obs)
        action_fn(obs)
        with pytest.raises(IndexError):
            action_fn(obs)


class TestReplayTrajectoryToPairedBatch:
    def test_shapes_and_row_counts_match(self):
        env = _StubEnv(num_envs=2)
        raw_actions = torch.randn(5, NUM_ACTIONS)
        obs, actions = replay_trajectory_to_paired_batch(env, raw_actions)
        assert obs.shape == (5 * 2, OBS_DIM)
        assert actions.shape == (5 * 2, NUM_ACTIONS)
        assert env.step_calls == 5
        assert env.reset_calls == 1

    def test_actions_correctly_paired_with_each_steps_obs(self):
        """The action tensor's row i must be the SAME raw_actions row that
        produced observation row i (env.num_envs=1 case: a 1:1 pairing, easy
        to check exactly)."""
        env = _StubEnv(num_envs=1)
        raw_actions = torch.arange(4 * NUM_ACTIONS, dtype=torch.float32).view(4, NUM_ACTIONS)
        obs, actions = replay_trajectory_to_paired_batch(env, raw_actions)
        assert obs.shape[0] == 4
        assert torch.allclose(actions, raw_actions)

    def test_multi_env_repeat_interleave_pairing(self):
        env = _StubEnv(num_envs=3)
        raw_actions = torch.arange(2 * NUM_ACTIONS, dtype=torch.float32).view(2, NUM_ACTIONS)
        obs, actions = replay_trajectory_to_paired_batch(env, raw_actions)
        assert actions.shape == (6, NUM_ACTIONS)
        # step 0's row repeated for the first 3 (num_envs) rows, step 1's for the next 3.
        assert torch.allclose(actions[0:3], raw_actions[0].unsqueeze(0).expand(3, -1))
        assert torch.allclose(actions[3:6], raw_actions[1].unsqueeze(0).expand(3, -1))


class TestPoolTrajectoryBatches:
    def test_concatenates_in_order(self):
        obs_a, actions_a = torch.zeros(3, OBS_DIM), torch.zeros(3, NUM_ACTIONS)
        obs_b, actions_b = torch.ones(2, OBS_DIM), torch.ones(2, NUM_ACTIONS)
        pooled_obs, pooled_actions = pool_trajectory_batches([(obs_a, actions_a), (obs_b, actions_b)])
        assert pooled_obs.shape == (5, OBS_DIM)
        assert pooled_actions.shape == (5, NUM_ACTIONS)
        assert torch.allclose(pooled_obs[:3], obs_a)
        assert torch.allclose(pooled_obs[3:], obs_b)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            pool_trajectory_batches([])


class TestGetDefaultArmJointPos:
    def test_reads_correct_slice(self):
        default_full = torch.arange(9, dtype=torch.float32).unsqueeze(0)  # (1, 9)
        env = _StubEnv(num_envs=1, default_joint_pos=default_full)
        result = get_default_arm_joint_pos(env)
        assert torch.allclose(result, torch.arange(7, dtype=torch.float32))


class TestBcPretrainUntilPlateau:
    def _stub_actor_critic(self):
        class _StubActorCritic(torch.nn.Module):
            def __init__(self, obs_dim, num_actions):
                super().__init__()
                self.actor = torch.nn.Linear(obs_dim, num_actions)
                with torch.no_grad():
                    self.actor.weight.zero_()
                    self.actor.bias.zero_()

            def act_inference(self, obs):
                return self.actor(obs["policy"])

        return _StubActorCritic(OBS_DIM, NUM_ACTIONS)

    def test_stops_early_when_plateaued(self):
        torch.manual_seed(0)
        obs = torch.randn(40, OBS_DIM)
        actions = torch.full((40, NUM_ACTIONS), 3.0)
        student = self._stub_actor_critic()
        optimizer = torch.optim.Adam(student.actor.parameters(), lr=0.5)
        gen = torch.Generator(device="cpu").manual_seed(0)

        loss_history, stopped_early = bc_pretrain_until_plateau(
            obs, actions, student, optimizer, batch_size=8, epochs_per_round=2, max_rounds=50, plateau_window=3, plateau_rel_tol=0.02, generator=gen
        )
        assert stopped_early
        assert len(loss_history) < 50
        assert loss_history[-1] < loss_history[0]

    def test_runs_to_max_rounds_when_plateau_never_satisfied(self):
        torch.manual_seed(0)
        obs = torch.randn(20, OBS_DIM)
        actions = torch.full((20, NUM_ACTIONS), 1.0)
        student = self._stub_actor_critic()
        optimizer = torch.optim.Adam(student.actor.parameters(), lr=0.05)
        gen = torch.Generator(device="cpu").manual_seed(0)

        # An impossible-to-satisfy relative-improvement bar forces the loop
        # to run the full max_rounds every time.
        loss_history, stopped_early = bc_pretrain_until_plateau(
            obs, actions, student, optimizer, batch_size=4, epochs_per_round=1, max_rounds=5, plateau_window=2, plateau_rel_tol=-999.0, generator=gen
        )
        assert not stopped_early
        assert len(loss_history) == 5

    def test_max_rounds_below_one_raises(self):
        student = self._stub_actor_critic()
        optimizer = torch.optim.Adam(student.actor.parameters(), lr=0.01)
        with pytest.raises(ValueError):
            bc_pretrain_until_plateau(torch.randn(4, OBS_DIM), torch.randn(4, NUM_ACTIONS), student, optimizer, batch_size=2, epochs_per_round=1, max_rounds=0)


class TestMainRejectsMultiShapeRealRun:
    """Regression guard for a real bug found on this task's own second real-
    GPU dispatch: running d8 then d10 sequentially in ONE process (build
    env, replay+BC-train, close, build the next shape's env the same way)
    doesn't crash but hangs the SECOND shape's env for 45+ minutes of
    severely degraded activity - the same underlying Isaac Lab
    single-simulation-context limitation `tasks/franka/distillation.py`'s
    own module docstring already documents as a crash for the
    simultaneous-envs case. `main`'s real (non-dry-run) branch must reject
    more than one shape immediately, before ever touching Isaac Sim, rather
    than silently hanging a future dispatch."""

    def _namespace(self, **overrides):
        defaults = dict(
            shapes="d8,d10",
            dry_run=False,
            seed=0,
            output_dir="/tmp/does-not-matter",
            trajectory_dir="/tmp/does-not-matter",
            device="cpu",
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_real_run_with_multiple_shapes_raises_before_touching_isaac_sim(self, tmp_path):
        args = self._namespace(shapes="d8,d10", output_dir=str(tmp_path))
        with pytest.raises(ValueError, match="exactly ONE shape"):
            main(args)

    def test_real_run_with_single_shape_does_not_raise_this_particular_error(self, tmp_path):
        """Single-shape real runs must get past the guard (and fail later,
        for the mundane reason that no real trajectory dir/Isaac Sim exists
        in this CPU-only test environment - NOT the multi-shape guard)."""
        args = self._namespace(shapes="d8", output_dir=str(tmp_path), trajectory_dir=str(tmp_path / "nonexistent"))
        with pytest.raises(FileNotFoundError):
            main(args)


class TestBcPretrainShape:
    def test_end_to_end_against_stub_env_and_trajectories(self):
        torch.manual_seed(0)
        env = _StubEnv(num_envs=1, obs_dim=41)
        trajectories = [_make_trajectory(num_steps=6, seed=s) for s in range(3)]
        gen = torch.Generator(device="cpu").manual_seed(0)

        student, loss_history, stopped_early = bc_pretrain_shape(
            env, trajectories, device="cpu", batch_size=4, epochs_per_round=2, max_rounds=6, plateau_window=2, plateau_rel_tol=-999.0, generator=gen
        )
        assert len(loss_history) == 6
        assert not stopped_early
        assert hasattr(student, "act_inference")
        # 3 trajectories x 6 steps x 1 env each replayed and pooled - env
        # visited 18 total steps across all 3 replays.
        assert env.step_calls == 18
        assert env.reset_calls == 3

    def test_multiple_trajectories_actually_pooled_not_just_last(self):
        """A regression guard against silently dropping all but the last
        trajectory: with trajectories of DIFFERENT lengths, the pooled
        dataset regress_on_paired_batches sees must reflect ALL of them."""
        torch.manual_seed(0)
        env = _StubEnv(num_envs=1, obs_dim=41)
        trajectories = [_make_trajectory(num_steps=n, seed=n) for n in (4, 7, 5)]
        gen = torch.Generator(device="cpu").manual_seed(0)

        bc_pretrain_shape(env, trajectories, device="cpu", batch_size=4, epochs_per_round=1, max_rounds=1, generator=gen)
        assert env.step_calls == 4 + 7 + 5
        assert env.reset_calls == 3
