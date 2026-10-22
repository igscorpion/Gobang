# A2C research web integration: planning only

No source, model, manifest, running service, game or training/evaluation queue was changed. No final-test outcomes were read to select a model. Implementation remains gated on all final tests and their independent audit.

## Fixed model to expose

- Suggested independent name: `a2c_residual10_guarded`.
- Checkpoint: `runs/a2c_research_20261020/confirmation/residual_tactical10_seed11/train/candidates/episode_030000.pt`.
- SHA256: `dea2fee0d128485ed20461f7ce5e793f4a9a193870af068ee4a03c9940c5b898`.
- Chosen from development before final tests: seed 11, episode 30000, residual/tactical10, guarded wrapper.
- Inference should reproduce the frozen CPU recipe: `training=False`, no epsilon, TacticalConfig enabled/depth5/nodes1500/candidates12/cache4096. Do not silently inherit different `GOMOKU_A2C_TACTIC_*` or CUDA settings and present them as identical tested inference.
- Keep `a2c4`, `a2c4_tactical`, their checkpoint, existing JSON files and historical reports.

## Existing refresh behavior

`rl/agents/registry.py:119/128/177` calls `_local_models()` on every list/create request. It rereads only `runs/production/models.json` and `a2c_models.json`; JSON metadata is not cached. Existing recognized model types can therefore be updated without reloading Python, and the selected existing agent is not replaced until a create/set action.

The current factory recognizes AlphaZero, standard four-channel A2C, and otherwise falls back to DQN. A new 10-channel residual checkpoint cannot be installed by JSON alone: standard A2CAgent.load rejects it. Do not disguise it as the existing `a2c` type.

`web/backend/main.py:18` binds registry functions on startup. `/game/agents` (line122) calls the list function each request; `/game/agent` (line127) creates an agent and resets its current game. `/game/rules` (line145) updates the process-local rule setting, reconstructs the current agent, and resets the game. The backend has no code-reload or generic plugin registration API.

`web/frontend/src/App.vue:87/101` fetches the names only on mount/bootstrap. Bootstrap also calls POST `/game/start`, so refreshing the browser resets the shared board. Its existing reset button does not reread names. `src/api/game.js:45` already exports GET-only `listAgents()`.

## Minimum implementation, after the final audit

1. `rl/agents/registry.py`: add an optional independent `a2c_research_models.json` to the dynamic metadata merge, and add an explicit `a2c_research` factory branch. Checkpoint must remain inside PROJECT_ROOT, exist, and match the recorded SHA. Load ModelSpec from its weights-only metadata; instantiate/load ResearchA2CAgent under the checkpoint's training rule first, then explicitly set the requested current rule mode and wrap with GuardedA2CAgent. Loading a freestyle model for renju must be explicit rule transfer; the feature encoder and guard must both use the requested rule mode. Preserve old factory behavior and disallow duplicate new names that overwrite old entries.
2. New `runs/production/a2c_research_models.json`: one independent entry with algorithm `a2c_research`, wrapper `guarded`, CPU device, the fixed checkpoint path/SHA, development selection provenance, frozen manifest/report/audit links and the fixed wrapper config. Reference the immutable selected checkpoint; copying is unnecessary inside the same project. Do not rewrite old model JSON files or the checkpoint.
3. New `tests/test_a2c_research_registry.py`: verify old/new names coexist, SHA/path rejection, the actual model spec and unchanged bytes, rule transfer for both feature encoder and guard, and legal black/white renju fixed-position moves. These are integration checks, not another strength test or model-selection stage.
4. Optional `web/frontend/src/App.vue`: add GET-only `refreshAgents()` on an explicit refresh action or selector focus so new names appear without POST `/game/start`. The existing API helper suffices; no API path change is required. Without this optional edit, a normal page reload refreshes names and resets the board.

No backend logic or frozen inference module needs changing merely to pass observations to this guard: the backend already supplies the previous action when `needs_last_move=True`. The existing interface currently always uses human black and AI white; adding color selection is outside this minimal integration.

## Process reload limitation and preserving the old service

Dynamic JSON reads do not provide Python source hot-loading. The first addition of the new factory branch must be loaded into a Python service process. Editing registry.py cannot by itself update an already imported function. If a server uses uvicorn `--reload`, a source edit may restart its worker and reset its shared game; do not assume this is a zero-interruption update.

To keep an existing service untouched, after the audit use a separate deployment snapshot outside its confirmed watch directories, start the updated backend on a free alternate port, and point a separate frontend instance to it using existing `VITE_API_PORT` (`src/api/game.js:1`) on a free frontend port. Leave old PIDs, ports and game intact. A future ordinary startup of the main service can load the integrated registry. Do not use runtime injection/monkey-patching of an existing web process.

The current read-only GET probes to default addresses 127.0.0.1:8000/game/agents and :5173/ were refused. This does not establish whether another configured port/service exists. Process-command inspection was unavailable in this sandbox. Verify actual listeners, launcher flags and watch directories before the eventual deployment; no service was started or restarted here.

## Validation without touching a user's current game

- First run isolated registry tests and an offline actual-checkpoint construction/SHA/spec check. Use fixed diagnostic positions, not reserved final-test games.
- Use a separate backend process or isolated TestClient for GET list, POST agent selection and rule switching. Confirm all old names remain, the new name constructs the frozen guard, and the same checkpoint SHA remains unchanged. POST selection/rules intentionally resets only that isolated game's state.
- If the optional frontend refresh is added, build it and verify refresh sends GET `/game/agents` only. Existing live board state must not change just to refresh available names.
- Live nonmutating confirmation is GET `/game/agents`. Let the user choose the new model; do not silently swap the active opponent or POST to the old live game.
- Preserve CPU/framework and explicit wrapper configuration in the deployment metadata. `last_search_nodes=0` is uninstrumented for this guard; it must not be presented as no VCF/search.
