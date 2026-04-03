# Plugins Subsystem

Optional integrations. Removing this entire directory must not break anything.
`TestPluginsOptional` enforces this mechanically.

## I want to...

### Use Abel causal discovery
1. Run: `causal-edge discover <TICKER>`
2. Complete browser OAuth if prompted; key is stored in `.env`
3. Use `--mode parents` or `--mode mb` depending on the discovery need
4. Copy the output YAML into your `strategies.yaml`
5. No API key? Fill `parents:` manually — framework works identically

### Understand plugin isolation
- Framework uses `try/except ImportError` to detect plugins, not registry
- No plugin code is imported at framework startup
- Core tests pass with `plugins/` directory deleted

### Build a new plugin (future)
- Create `causal_edge/plugins/<name>/` directory
- Expose capabilities via top-level functions
- Framework discovers via `try/except` import in `causal_edge/cli.py`
- No registry pattern until second plugin exists (YAGNI)
