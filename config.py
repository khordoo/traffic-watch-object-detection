import os


class Config:

    def _env_var(self, key, default=None, required=False):
        value = os.getenv(key, default)
        if required and value is None:
            ValueError(f"The required environment variable is not set : {key}")
        return value

    @property
    def postgres_host(self):
        return self._env_var("POSTGRES_DATABASE_HOST", required=True)

    @property
    def postgres_database_name(self):
        return self._env_var("POSTGRES_DATABASE_NAME", required=True)

    @property
    def postgres_username(self):
        return self._env_var("POSTGRES_DATABASE_USERNAME", required=True)

    @property
    def postgres_password(self):
        return self._env_var("POSTGRES_DATABASE_PASSWORD", required=True)
