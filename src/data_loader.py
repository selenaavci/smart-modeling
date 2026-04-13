"""Dosya okuma katmanı — CSV, Excel ve XML desteği."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".xml"}


def load_dataframe(uploaded_file) -> pd.DataFrame:
    """Streamlit'ten gelen dosyayı pandas DataFrame olarak döner.

    Parameters
    ----------
    uploaded_file : UploadedFile
        `st.file_uploader` tarafından verilen nesne. `.name` ve binary içeriğe
        sahip olması yeterlidir.

    Returns
    -------
    pandas.DataFrame
    """
    name = getattr(uploaded_file, "name", "") or ""
    ext = Path(name).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Desteklenmeyen dosya formatı: {ext}. "
            f"Lütfen şunlardan birini yükleyin: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if ext == ".csv":
        return _read_csv_robust(uploaded_file)

    if ext in (".xlsx", ".xls"):
        return pd.read_excel(uploaded_file)

    if ext == ".xml":
        return _read_xml(uploaded_file)

    raise ValueError(f"Desteklenmeyen dosya formatı: {ext}")


def _read_csv_robust(uploaded_file) -> pd.DataFrame:
    """CSV dosyalarını yaygın ayırıcı/encoding varyasyonlarını deneyerek okur."""
    attempts = [
        {"sep": ","},
        {"sep": ";"},
        {"sep": "\t"},
        {"sep": ",", "encoding": "latin-1"},
        {"sep": ";", "encoding": "latin-1"},
    ]
    last_err: Exception | None = None
    for opts in attempts:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, **opts)
            if df.shape[1] >= 1:
                return df
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    if last_err:
        raise last_err
    raise ValueError("CSV dosyası okunamadı.")


def _read_xml(uploaded_file) -> pd.DataFrame:
    """XML dosyasını tabular hale getirir.

    `pandas.read_xml` tekrar eden aynı seviyedeki elementleri satır olarak
    alır. Karmaşık iç içe XML yapılarında kullanıcıya anlamlı bir hata verir.
    """
    try:
        uploaded_file.seek(0)
        return pd.read_xml(uploaded_file)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            "XML dosyası tabular yapıya dönüştürülemedi. "
            "Dosyanın tekrar eden aynı seviyedeki kayıtlardan oluştuğundan "
            "emin olun (örn. <rows><row>...</row><row>...</row></rows>). "
            f"Detay: {exc}"
        ) from exc
