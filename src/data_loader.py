"""Dosya okuma katmanı — CSV, Excel ve XML desteği."""
from __future__ import annotations

import io
import xml.etree.ElementTree as ET
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
        except Exception as exc:
            last_err = exc
    if last_err:
        raise last_err
    raise ValueError("CSV dosyası okunamadı.")


def _read_xml(uploaded_file) -> pd.DataFrame:
    """XML dosyasını tabular hale getirir.

    Düz XML'lerin yanı sıra <records><record>...</record></records> gibi iç içe
    yapıları da doğru okur; alt elemanlar `originator_name`, `beneficiary_iban`
    gibi düzleştirilmiş sütunlara açılır.
    """
    if hasattr(uploaded_file, "seek"):
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
    raw = uploaded_file.read() if hasattr(uploaded_file, "read") else uploaded_file
    if isinstance(raw, str):
        raw = raw.encode("utf-8")

    def _strip_ns(tag: str) -> str:
        return tag.split("}")[-1] if "}" in tag else tag

    def _flatten(elem, parent_key: str = "") -> dict:
        row: dict = {}
        for k, v in elem.attrib.items():
            row[f"{parent_key}_@{k}" if parent_key else f"@{k}"] = v
        children = list(elem)
        if not children:
            text = elem.text.strip() if elem.text and elem.text.strip() else None
            if parent_key:
                row[parent_key] = text
            return row
        for child in children:
            tag = _strip_ns(child.tag)
            new_key = f"{parent_key}_{tag}" if parent_key else tag
            row.update(_flatten(child, new_key))
        return row

    try:
        root = ET.fromstring(raw)
        records = root.findall(".//record")
        if records:
            df = pd.DataFrame([_flatten(r) for r in records])
            if not df.empty:
                return df
    except ET.ParseError:
        pass

    last_err: Exception | None = None
    for parser in ("lxml", "etree"):
        try:
            df = pd.read_xml(io.BytesIO(raw), parser=parser)
            if df is not None and not df.empty:
                return df
        except ImportError:
            continue
        except Exception as exc:
            last_err = exc

    raise ValueError(
        "XML dosyası tabular yapıya dönüştürülemedi. "
        "Dosyanın tekrar eden aynı seviyedeki kayıtlardan oluştuğundan "
        "emin olun (örn. <rows><row>...</row><row>...</row></rows>). "
        f"Detay: {last_err}"
    ) from last_err
