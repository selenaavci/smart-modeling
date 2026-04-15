from __future__ import annotations

from io import BytesIO
from typing import Dict

import pandas as pd


def build_excel_report(sheets: Dict[str, pd.DataFrame]) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe = name[:31]
            df.to_excel(writer, sheet_name=safe, index=False)
    return buffer.getvalue()
