# Config management

Our configuration management is done through dynaconf. This means there are a variety of ways to update a variable.
The most obvious is settings.toml, a default of which is in scripts. Whenever a python module loads config.py, it
will attempt to find a file called settings.toml relative to the python module entrypoint (the one with main). If not
found, it will recurse through local directories until it does.

Configs can also be done through envvars, prefixed with TRANSFERWARE. Ex: `TRANSFERWARE_TRAINING.EPOCHS`.

For reference on what can be configured, please reference the demo config in scripts. This config will be loaded by
the application scripts by default. One can make another file alongside it called `settings.local.toml` to contact
machine specific files, which will override the defaults. Because this is alongside the scripts, this will serve
as the main config for both the API and the training job.