from dynaconf import Dynaconf

# Attempt to load a settings file *relative to the entrypoint python module*
settings = Dynaconf(
    envvar_prefix="TRANSFERWARE",
    settings_files=['settings.toml'],
)
