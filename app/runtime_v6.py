"""V6 paper profile; the financial policy and signed RuntimeSettings stay fixed."""
from typing import Literal

from pydantic import model_validator

from app.runtime_settings import RuntimeSettings


class RuntimeV6Settings(RuntimeSettings):
    volium_strategy_profile: Literal['v1_guarded', 'v5_strict', 'v5_equal', 'v6_current_leg'] = 'v6_current_leg'

    @model_validator(mode='after')
    def validate_v6_paper_profile(self):
        if self.volium_strategy_profile == 'v6_current_leg':
            if self.auto_execution:
                raise ValueError('Experimental V6 supports paper trading only')
            if self.volium_mode != 'intraday':
                raise ValueError('V6 supports the D1/H1/M5 intraday model only')
        return self
