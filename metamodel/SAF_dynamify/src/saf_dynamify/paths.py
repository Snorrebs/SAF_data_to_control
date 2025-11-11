from pathlib import Path
import yaml


class Config:
    def __init__(self, path: str | Path = "config.yaml"):
        with open(path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)


    def p(self, *keys):
        val = self.cfg
        for k in keys:
            val = val[k]
        return val


    @property
    def raw_csv(self) -> Path:
        return Path(self.p("paths", "raw_csv"))


    @property
    def interim_csv(self) -> Path:
        return Path(self.p("paths", "interim_csv"))


    @property
    def processed_parquet(self) -> Path:
        return Path(self.p("paths", "processed_parquet"))


    @property
    def figs_dir(self) -> Path:
        return Path(self.p("paths", "figs_dir"))


    @property
    def models_dir(self) -> Path:
        return Path(self.p("paths", "models_dir"))