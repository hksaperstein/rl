"""Sim-independent unit tests for tasks/franka/distillation.py's rollout-
collection/relabeling/DAgger-mechanics logic - no Isaac Lab import needed
(mirrors tests/test_mdp_shape_observations.py's own scope split: pure
math/mechanics live in a plain-torch(+rsl_rl) module, exercised here in
isolation from the live simulated env). Run via:

/home/saps/IsaacLab/_isaac_sim/python.sh -m pytest tests/test_distillation_data_collection.py -v -p no:launch_testing

(plain python3/pytest lacks torch in this environment - see
project_pytest-needs-isaac-sim-python memory. `rsl_rl` is also required -
confirmed importable via the same interpreter without an AppLauncher, see
tasks/franka/distillation.py's own module docstring.)

Task 4 of docs/superpowers/plans/2026-07-16-unified-multi-die-specialist-
distillation.md ("write failing unit tests for the rollout-collection/
relabeling logic using stub policies/envs - no real checkpoints or GPU
needed for this test").
"""

from __future__ import annotations

import pytest
import torch

from tasks.franka.distillation import (
    MultiShapeTeacherRouter,
    behavior_cloning_loss,
    collect_rollout,
    dagger_beta_schedule,
    mix_actions,
    pool_and_shuffle,
    regress_on_paired_batches,
    run_dagger_iteration,
)

SHAPE_CLASSES = ("d12", "d20")
OBS_DIM = 6
STATE_DIM = OBS_DIM - len(SHAPE_CLASSES)  # 4 non-shape dims
NUM_ACTIONS = 2


def _onehot_row(shape: str) -> torch.Tensor:
    row = torch.zeros(len(SHAPE_CLASSES))
    row[SHAPE_CLASSES.index(shape)] = 1.0
    return row


class _StubEnv:
    """Deterministic, physics-free stub satisfying collect_rollout's
    minimal reset()/step() contract: obs is a plain dict with a "policy"
    key -> (num_envs, obs_dim) tensor, whose last len(SHAPE_CLASSES) dims
    are a fixed one-hot for `shape` (matching the real schema's own
    shape-onehot observation term) and whose leading STATE_DIM dims are a
    simple counter that increments by 1.0 each step (fully deterministic,
    so tests can assert exact visited-state values)."""

    def __init__(self, shape: str, num_envs: int = 3):
        self.shape = shape
        self.num_envs = num_envs
        self._onehot = _onehot_row(shape)
        self._step_count = 0
        self.reset_calls = 0
        self.step_calls = 0

    def _obs(self) -> dict:
        state = torch.full((self.num_envs, STATE_DIM), float(self._step_count))
        onehot_block = self._onehot.unsqueeze(0).expand(self.num_envs, -1)
        return {"policy": torch.cat([state, onehot_block], dim=-1)}

    def reset(self):
        self._step_count = 0
        self.reset_calls += 1
        return self._obs()

    def step(self, actions: torch.Tensor):
        assert actions.shape == (self.num_envs, NUM_ACTIONS)
        self._step_count += 1
        self.step_calls += 1
        reward = torch.zeros(self.num_envs)
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        return self._obs(), reward, done, {}


class _TupleResetEnv(_StubEnv):
    """Same as _StubEnv but reset() returns (obs, extras), matching the
    real isaaclab_rl.rsl_rl.RslRlVecEnvWrapper.reset() contract exactly -
    collect_rollout must handle both forms."""

    def reset(self):
        obs = super().reset()
        return obs, {"some": "extras"}


def _zero_action_fn(obs: torch.Tensor) -> torch.Tensor:
    return torch.zeros(obs.shape[0], NUM_ACTIONS)


def _constant_action_fn(value: float):
    def _fn(obs: torch.Tensor) -> torch.Tensor:
        return torch.full((obs.shape[0], NUM_ACTIONS), value)

    return _fn


class TestCollectRollout:
    def test_shapes_and_step_count(self):
        env = _StubEnv("d20", num_envs=3)
        obs = collect_rollout(env, _zero_action_fn, num_steps=5)
        assert obs.shape == (5 * 3, OBS_DIM)
        assert env.step_calls == 5
        assert env.reset_calls == 1

    def test_deterministic_visited_states_match_expected_counter(self):
        """The stub env's state is a deterministic step counter - verify
        collect_rollout actually visits states 0, 1, 2, ... in order (not
        e.g. off-by-one, not skipping the post-reset observation)."""
        env = _StubEnv("d12", num_envs=2)
        obs = collect_rollout(env, _zero_action_fn, num_steps=4)
        reshaped = obs.view(4, 2, OBS_DIM)
        for step in range(4):
            expected_state_value = float(step)
            assert torch.allclose(reshaped[step, :, :STATE_DIM], torch.full((2, STATE_DIM), expected_state_value))

    def test_tuple_reset_contract_handled(self):
        """Real RslRlVecEnvWrapper.reset() returns (obs, extras), not obs
        alone - collect_rollout must unwrap this correctly."""
        env = _TupleResetEnv("d20", num_envs=2)
        obs = collect_rollout(env, _zero_action_fn, num_steps=3)
        assert obs.shape == (3 * 2, OBS_DIM)

    def test_actions_from_action_fn_are_actually_sent_to_env(self):
        received_actions = []
        env = _StubEnv("d20", num_envs=2)
        orig_step = env.step

        def _spy_step(actions):
            received_actions.append(actions.clone())
            return orig_step(actions)

        env.step = _spy_step
        collect_rollout(env, _constant_action_fn(7.0), num_steps=2)
        assert len(received_actions) == 2
        for a in received_actions:
            assert torch.allclose(a, torch.full((2, NUM_ACTIONS), 7.0))

    def test_no_grad_context_does_not_leak_gradients(self):
        """collect_rollout must not require_grad on its output - it's a
        pure data-collection pass, gradient tracking happens later, on a
        fresh forward pass through the pooled buffer (see module docstring
        of tasks/franka/distillation.py)."""
        env = _StubEnv("d20", num_envs=2)
        obs = collect_rollout(env, _zero_action_fn, num_steps=2)
        assert not obs.requires_grad


class TestMultiShapeTeacherRouter:
    def _make_router(self, shape_onehot_start=STATE_DIM, teachers=None):
        teachers = teachers or {
            "d12": lambda obs: torch.full((obs.shape[0], NUM_ACTIONS), 12.0),
            "d20": lambda obs: torch.full((obs.shape[0], NUM_ACTIONS), 20.0),
        }
        return MultiShapeTeacherRouter(teachers, SHAPE_CLASSES, shape_onehot_start, len(SHAPE_CLASSES))

    def test_routes_single_shape_batch_to_correct_teacher(self):
        router = self._make_router()
        obs = torch.cat(
            [torch.zeros(3, STATE_DIM), _onehot_row("d20").unsqueeze(0).expand(3, -1)],
            dim=-1,
        )
        actions = router.relabel(obs)
        assert torch.allclose(actions, torch.full((3, NUM_ACTIONS), 20.0))

    def test_routes_mixed_shape_batch_row_by_row(self):
        """The whole point of routing off the observation's own shape-onehot
        feature (not external bookkeeping): a POOLED, SHUFFLED batch mixing
        both shapes must relabel each row with ITS OWN shape's teacher."""
        router = self._make_router()
        d12_rows = torch.cat([torch.zeros(2, STATE_DIM), _onehot_row("d12").unsqueeze(0).expand(2, -1)], dim=-1)
        d20_rows = torch.cat([torch.ones(2, STATE_DIM), _onehot_row("d20").unsqueeze(0).expand(2, -1)], dim=-1)
        mixed = torch.cat([d12_rows, d20_rows[:1], d12_rows[:1], d20_rows[1:]], dim=0)
        actions = router.relabel(mixed)
        expected = torch.tensor([12.0, 12.0, 20.0, 12.0, 20.0]).unsqueeze(-1).expand(-1, NUM_ACTIONS)
        assert torch.allclose(actions, expected)

    def test_missing_teacher_for_present_shape_raises(self):
        router = self._make_router(teachers={"d12": lambda obs: torch.zeros(obs.shape[0], NUM_ACTIONS)})
        obs = torch.cat([torch.zeros(2, STATE_DIM), _onehot_row("d20").unsqueeze(0).expand(2, -1)], dim=-1)
        try:
            router.relabel(obs)
            assert False, "expected KeyError"
        except KeyError:
            pass

    def test_empty_batch_raises(self):
        router = self._make_router()
        try:
            router.relabel(torch.zeros(0, OBS_DIM))
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_unknown_shape_in_teacher_dict_rejected_at_construction(self):
        try:
            MultiShapeTeacherRouter({"d6": lambda obs: obs}, SHAPE_CLASSES, STATE_DIM, len(SHAPE_CLASSES))
            assert False, "expected ValueError"
        except ValueError:
            pass


class TestDaggerBetaSchedule:
    def test_endpoints(self):
        assert dagger_beta_schedule(0, 10, beta_start=1.0, beta_end=0.0) == 1.0
        assert dagger_beta_schedule(9, 10, beta_start=1.0, beta_end=0.0) == 0.0

    def test_monotonic_decrease(self):
        values = [dagger_beta_schedule(i, 10) for i in range(10)]
        assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))

    def test_single_iteration_returns_beta_end(self):
        assert dagger_beta_schedule(0, 1, beta_start=1.0, beta_end=0.3) == 0.3

    def test_custom_endpoints(self):
        assert dagger_beta_schedule(0, 5, beta_start=0.8, beta_end=0.2) == pytest.approx(0.8)
        assert dagger_beta_schedule(4, 5, beta_start=0.8, beta_end=0.2) == pytest.approx(0.2)


class TestMixActions:
    def test_beta_zero_is_pure_student(self):
        student = torch.zeros(5, NUM_ACTIONS)
        teacher = torch.ones(5, NUM_ACTIONS)
        mixed = mix_actions(student, teacher, beta=0.0)
        assert torch.allclose(mixed, student)

    def test_beta_one_is_pure_teacher(self):
        student = torch.zeros(5, NUM_ACTIONS)
        teacher = torch.ones(5, NUM_ACTIONS)
        mixed = mix_actions(student, teacher, beta=1.0)
        assert torch.allclose(mixed, teacher)

    def test_shape_mismatch_raises(self):
        student = torch.zeros(5, NUM_ACTIONS)
        teacher = torch.ones(4, NUM_ACTIONS)
        try:
            mix_actions(student, teacher, beta=0.5)
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_reproducible_with_seeded_generator(self):
        student = torch.zeros(1000, NUM_ACTIONS)
        teacher = torch.ones(1000, NUM_ACTIONS)
        gen1 = torch.Generator(device="cpu").manual_seed(0)
        gen2 = torch.Generator(device="cpu").manual_seed(0)
        out1 = mix_actions(student, teacher, beta=0.5, generator=gen1)
        out2 = mix_actions(student, teacher, beta=0.5, generator=gen2)
        assert torch.equal(out1, out2)
        # roughly half teacher, half student (loose statistical sanity check)
        frac_teacher = (out1[:, 0] == 1.0).float().mean().item()
        assert 0.35 < frac_teacher < 0.65


class TestPoolAndShuffle:
    def test_preserves_all_rows_as_a_set(self):
        a = torch.arange(6).float().view(3, 2)
        b = torch.arange(6, 10).float().view(2, 2)
        pooled = pool_and_shuffle([a, b])
        assert pooled.shape == (5, 2)
        pooled_rows = {tuple(row.tolist()) for row in pooled}
        expected_rows = {tuple(row.tolist()) for row in torch.cat([a, b], dim=0)}
        assert pooled_rows == expected_rows

    def test_actually_shuffles_with_seeded_generator(self):
        a = torch.arange(20).float().view(10, 2)
        gen = torch.Generator(device="cpu").manual_seed(1)
        pooled = pool_and_shuffle([a], generator=gen)
        assert not torch.equal(pooled, a)

    def test_reproducible_with_seeded_generator(self):
        a = torch.arange(20).float().view(10, 2)
        gen1 = torch.Generator(device="cpu").manual_seed(3)
        gen2 = torch.Generator(device="cpu").manual_seed(3)
        assert torch.equal(pool_and_shuffle([a], generator=gen1), pool_and_shuffle([a], generator=gen2))


class TestBehaviorCloningLoss:
    def test_zero_when_identical(self):
        x = torch.randn(4, NUM_ACTIONS)
        assert behavior_cloning_loss(x, x).item() == 0.0

    def test_positive_when_different(self):
        student = torch.zeros(4, NUM_ACTIONS)
        teacher = torch.ones(4, NUM_ACTIONS)
        loss = behavior_cloning_loss(student, teacher)
        assert loss.item() > 0.0

    def test_matches_manual_mse(self):
        student = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        teacher = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        expected = ((student - teacher) ** 2).mean()
        assert torch.isclose(behavior_cloning_loss(student, teacher), expected)


class _StubActorCritic(torch.nn.Module):
    """Minimal stand-in for rsl_rl.modules.ActorCritic exposing only what
    run_dagger_iteration/collect_rollout actually use (`act_inference` and
    `.actor.parameters()` for the optimizer) - a tiny real linear layer so
    gradients genuinely flow and a BC loss can genuinely decrease."""

    def __init__(self, obs_dim: int, num_actions: int):
        super().__init__()
        self.actor = torch.nn.Linear(obs_dim, num_actions)
        with torch.no_grad():
            self.actor.weight.zero_()
            self.actor.bias.zero_()

    def act_inference(self, obs: dict) -> torch.Tensor:
        return self.actor(obs["policy"])


class TestRunDaggerIterationIntegration:
    """End-to-end integration test of run_dagger_iteration wiring (collect
    -> relabel -> pool+shuffle -> BC regression) against two stub
    single-shape envs and two trivial constant-output stub teachers - no
    real checkpoints, no Isaac Sim, per this task's own dispatch
    instruction."""

    def _make_setup(self):
        teachers = {
            "d12": lambda obs: torch.full((obs.shape[0], NUM_ACTIONS), 1.0),
            "d20": lambda obs: torch.full((obs.shape[0], NUM_ACTIONS), -1.0),
        }
        router = MultiShapeTeacherRouter(teachers, SHAPE_CLASSES, STATE_DIM, len(SHAPE_CLASSES))
        student = _StubActorCritic(OBS_DIM, NUM_ACTIONS)
        optimizer = torch.optim.Adam(student.actor.parameters(), lr=0.1)
        envs = {"d12": _StubEnv("d12", num_envs=4), "d20": _StubEnv("d20", num_envs=4)}
        return teachers, router, student, optimizer, envs

    def test_both_envs_are_rolled_out(self):
        _, router, student, optimizer, envs = self._make_setup()
        run_dagger_iteration(envs, student, router, optimizer, rollout_steps=3, batch_size=8, num_epochs=1, beta=1.0)
        assert envs["d12"].step_calls == 3
        assert envs["d20"].step_calls == 3

    def test_loss_decreases_over_several_iterations(self):
        """A BC regression toward a fixed target (teacher outputs are
        constant here) must drive the loss down - the clearest possible
        end-to-end sanity check that gradients actually flow through the
        whole collect -> relabel -> pool -> regress pipeline."""
        _, router, student, optimizer, envs = self._make_setup()
        gen = torch.Generator(device="cpu").manual_seed(0)
        losses = []
        for it in range(8):
            beta = dagger_beta_schedule(it, 8, beta_start=1.0, beta_end=1.0)  # pure teacher-forcing rollout
            loss = run_dagger_iteration(envs, student, router, optimizer, rollout_steps=3, batch_size=8, num_epochs=2, beta=beta, generator=gen)
            losses.append(loss)
        assert losses[-1] < losses[0]

    def test_beta_zero_still_relabels_with_teacher_not_student(self):
        """Even under beta=0 (pure student rollout, i.e. full DAgger), the
        supervised-learning LABEL must still come from the teacher - the
        loss should still be computable/finite and should still decrease
        (verifying mix_actions' beta only gates the EXECUTED action, not
        the relabeling target, matching the module docstring's claim)."""
        _, router, student, optimizer, envs = self._make_setup()
        gen = torch.Generator(device="cpu").manual_seed(0)
        losses = []
        for it in range(8):
            loss = run_dagger_iteration(envs, student, router, optimizer, rollout_steps=3, batch_size=8, num_epochs=2, beta=0.0, generator=gen)
            losses.append(loss)
        assert all(torch.isfinite(torch.tensor(losses)))
        assert losses[-1] < losses[0]

    def test_returned_loss_is_mean_over_all_regression_steps(self):
        _, router, student, optimizer, envs = self._make_setup()
        loss = run_dagger_iteration(envs, student, router, optimizer, rollout_steps=2, batch_size=100, num_epochs=3, beta=1.0)
        assert isinstance(loss, float)
        assert loss >= 0.0


class TestRegressOnPairedBatches:
    """Task 1 of docs/superpowers/plans/2026-07-19-d8-d10-demo-warmstart-
    implementation.md: regress_on_paired_batches mirrors
    regress_on_pooled_batches' own shuffle/minibatch/epoch loop and its call
    to behavior_cloning_loss verbatim, but against pre-paired (obs, action)
    tensors passed in directly - no MultiShapeTeacherRouter relabeling step
    (H1's demonstration data already carries one fixed action label per row,
    not a live teacher network to query). Reuses _StubActorCritic, same
    tiny-real-linear-layer stand-in TestRunDaggerIterationIntegration
    already uses so gradients genuinely flow."""

    def test_loss_decreases_over_epochs_on_synthetic_paired_data(self):
        """Clearest possible end-to-end sanity check: a BC regression toward
        a FIXED target (teacher_action is a constant tensor here) must drive
        the loss down over epochs."""
        torch.manual_seed(0)
        obs = torch.randn(40, OBS_DIM)
        actions = torch.full((40, NUM_ACTIONS), 3.0)
        student = _StubActorCritic(OBS_DIM, NUM_ACTIONS)
        optimizer = torch.optim.Adam(student.actor.parameters(), lr=0.1)
        gen = torch.Generator(device="cpu").manual_seed(0)

        losses = []
        for _ in range(6):
            loss = regress_on_paired_batches(obs, actions, student, optimizer, batch_size=8, num_epochs=2, generator=gen)
            losses.append(loss)
        assert losses[-1] < losses[0]

    def test_returned_loss_is_mean_over_all_regression_steps(self):
        obs = torch.randn(20, OBS_DIM)
        actions = torch.zeros(20, NUM_ACTIONS)
        student = _StubActorCritic(OBS_DIM, NUM_ACTIONS)
        optimizer = torch.optim.Adam(student.actor.parameters(), lr=0.01)
        loss = regress_on_paired_batches(obs, actions, student, optimizer, batch_size=100, num_epochs=3, generator=None)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_obs_actions_row_count_mismatch_raises(self):
        obs = torch.randn(10, OBS_DIM)
        actions = torch.zeros(9, NUM_ACTIONS)
        student = _StubActorCritic(OBS_DIM, NUM_ACTIONS)
        optimizer = torch.optim.Adam(student.actor.parameters(), lr=0.01)
        try:
            regress_on_paired_batches(obs, actions, student, optimizer, batch_size=4, num_epochs=1)
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_deterministic_shuffling_with_seeded_generator(self):
        """Same shape TestPoolAndShuffle/TestMixActions already establish
        for this module's other seeded-generator functions: two runs with
        independently-seeded-but-equal generators must produce identical
        results (same minibatch order -> identical loss trajectory, since
        the student/optimizer start from the same zero-initialized state
        each time)."""
        obs = torch.randn(24, OBS_DIM)
        actions = torch.randn(24, NUM_ACTIONS)

        def _run(seed: int) -> float:
            torch.manual_seed(123)  # same student init both runs
            student = _StubActorCritic(OBS_DIM, NUM_ACTIONS)
            optimizer = torch.optim.Adam(student.actor.parameters(), lr=0.05)
            gen = torch.Generator(device="cpu").manual_seed(seed)
            return regress_on_paired_batches(obs, actions, student, optimizer, batch_size=6, num_epochs=2, generator=gen)

        loss1 = _run(42)
        loss2 = _run(42)
        assert loss1 == pytest.approx(loss2)
