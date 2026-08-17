from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import Draw, Fullscreen, MeasureControl
from shapely.geometry import shape, mapping
from shapely import wkt as shapely_wkt
from streamlit_folium import st_folium

try:
    import asf_search as asf
except Exception:
    asf = None

try:
    import hyp3_sdk as sdk
except Exception:
    sdk = None

try:
    import rasterio
    from rasterio.enums import ColorInterp
    from matplotlib.colors import hsv_to_rgb
    from scipy.ndimage import gaussian_filter
except Exception:
    rasterio = None


st.set_page_config(
    page_title="InSAR Explorer",
    page_icon="🛰️",
    layout="wide",
)

st.title("InSAR Explorer")
st.caption(
    "Búsqueda Sentinel-1 + descarga ASF + HyP3 On Demand + procesamiento local SNAP"
)


def init_state():
    defaults = {
        "aoi_geojson": None,
        "aoi_wkt": None,
        "search_results": None,
        "selected_granules": [],
        "last_hyp3_product_dir": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def geometry_to_wkt(geojson_geometry):
    geom = shape(geojson_geometry)
    if geom.geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError("El AOI debe ser un Polygon o MultiPolygon.")
    return geom.wkt


def extract_latest_polygon(map_state):
    if not map_state:
        return None

    drawings = map_state.get("all_drawings") or []
    if drawings:
        for item in reversed(drawings):
            geom = item.get("geometry")
            if geom and geom.get("type") in {"Polygon", "MultiPolygon"}:
                return geom

    last = map_state.get("last_active_drawing")
    if isinstance(last, dict):
        geom = last.get("geometry")
        if geom and geom.get("type") in {"Polygon", "MultiPolygon"}:
            return geom

    return None


def safe_props(product):
    try:
        p = product.properties
    except Exception:
        p = {}

    return {
        "sceneName": p.get("sceneName") or p.get("fileName") or p.get("granuleName"),
        "startTime": p.get("startTime"),
        "stopTime": p.get("stopTime"),
        "flightDirection": p.get("flightDirection"),
        "pathNumber": p.get("pathNumber"),
        "frameNumber": p.get("frameNumber"),
        "polarization": p.get("polarization"),
        "beamModeType": p.get("beamModeType"),
        "platform": p.get("platform"),
        "processingLevel": p.get("processingLevel"),
        "url": (product.get_urls()[0] if hasattr(product, "get_urls") and product.get_urls() else None),
    }


def make_asf_session(token: str):
    if asf is None:
        raise RuntimeError("Falta instalar asf_search.")
    token = token.strip()
    if not token:
        raise ValueError("Ingresa tu token de Earthdata.")
    return asf.ASFSession().auth_with_token(token)


def search_asf(aoi_wkt, start, end, direction, max_results):
    if asf is None:
        raise RuntimeError("Falta instalar asf_search.")

    kwargs = dict(
        platform=[asf.PLATFORM.SENTINEL1],
        intersectsWith=aoi_wkt,
        start=start.isoformat(),
        end=end.isoformat(),
        maxResults=int(max_results),
        beamMode="IW",
        processingLevel="SLC",
    )

    if direction != "AMBAS":
        kwargs["flightDirection"] = direction

    return asf.geo_search(**kwargs)


def normalize_granule_name(name: str) -> str:
    name = Path(name).name
    name = re.sub(r"\.SAFE$", "", name, flags=re.I)
    name = re.sub(r"\.zip$", "", name, flags=re.I)
    return name


def validate_pair_names(g1: str, g2: str):
    problems = []
    if not g1 or not g2:
        problems.append("Debes seleccionar dos escenas.")
        return problems

    p1 = g1.split("_")
    p2 = g2.split("_")

    if "_IW_SLC_" not in g1 or "_IW_SLC_" not in g2:
        problems.append("Las dos entradas deberían ser Sentinel-1 IW SLC.")

    if "1SDV" not in g1 and "1SSV" not in g1:
        problems.append("Revisa la polarización de la primera escena.")
    if "1SDV" not in g2 and "1SSV" not in g2:
        problems.append("Revisa la polarización de la segunda escena.")

    return problems


def run_process(cmd, cwd=None):
    process = subprocess.Popen(
        [str(x) for x in cmd],
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    lines = []
    while True:
        line = process.stdout.readline()
        if line:
            lines.append(line)
            yield line
        if line == "" and process.poll() is not None:
            break

    rc = process.wait()
    if rc != 0:
        raise RuntimeError("El proceso terminó con error.\n" + "".join(lines[-60:]))


def find_tiff(folder: Path, suffix: str):
    matches = list(folder.rglob(f"*{suffix}"))
    return matches[0] if matches else None


def extract_hyp3_zip(zip_path: Path, output_root: Path):
    product_name = zip_path.stem
    target = output_root / product_name
    target.mkdir(parents=True, exist_ok=True)

    marker = target / ".extracted"
    if not marker.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target)
        marker.write_text("ok", encoding="utf-8")

    nested = target / product_name
    return nested if nested.exists() else target


def create_color_phase(
    phase_tif: Path,
    coherence_tif: Path,
    amplitude_tif: Path,
    output_tif: Path,
    sigma: float = 1.0,
    coh_gray: float = 0.10,
    coh_color: float = 0.65,
):
    if rasterio is None:
        raise RuntimeError("Faltan rasterio/scipy/matplotlib.")

    with rasterio.open(phase_tif) as src:
        phase = src.read(1).astype(np.float32)
        mask_phase = src.read_masks(1) > 0
        profile = src.profile.copy()

    with rasterio.open(coherence_tif) as src:
        coherence = src.read(1).astype(np.float32)
        mask_coh = src.read_masks(1) > 0

    with rasterio.open(amplitude_tif) as src:
        amplitude = src.read(1).astype(np.float32)
        mask_amp = src.read_masks(1) > 0

    if not (phase.shape == coherence.shape == amplitude.shape):
        raise ValueError("Fase, coherencia y amplitud no tienen la misma grilla.")

    valid = (
        mask_phase & mask_amp & np.isfinite(phase) &
        np.isfinite(amplitude) & (amplitude > 0)
    )
    if not np.any(valid):
        raise RuntimeError("No hay píxeles válidos para crear Color Phase.")

    coh = np.where(
        mask_coh & np.isfinite(coherence),
        np.clip(coherence, 0, 1),
        0,
    ).astype(np.float32)

    weight = np.where(valid, np.maximum(coh, 0.02), 0).astype(np.float32)
    real = np.where(valid, np.cos(phase) * weight, 0).astype(np.float32)
    imag = np.where(valid, np.sin(phase) * weight, 0).astype(np.float32)

    opts = {"sigma": sigma, "mode": "constant", "cval": 0.0}
    real_f = gaussian_filter(real, **opts)
    imag_f = gaussian_filter(imag, **opts)
    w_f = np.maximum(gaussian_filter(weight, **opts), 1e-8)
    phase_f = np.arctan2(imag_f / w_f, real_f / w_f)

    hue = np.mod(phase_f + np.pi, 2 * np.pi) / (2 * np.pi)
    saturation = np.clip((coh - coh_gray) / max(coh_color - coh_gray, 1e-6), 0, 1)

    amp_log = np.full(amplitude.shape, np.nan, dtype=np.float32)
    amp_log[valid] = 20 * np.log10(np.maximum(amplitude[valid], 1e-8))
    lo, hi = np.nanpercentile(amp_log[valid], [2, 98])
    brightness = np.clip((amp_log - lo) / max(hi - lo, 1e-6), 0, 1)
    brightness = np.power(brightness, 1.1)
    brightness = brightness * (0.18 + 0.82 * saturation)

    hsv = np.stack([hue, saturation, np.clip(brightness, 0, 1)], axis=-1)
    rgb = np.clip(hsv_to_rgb(hsv) * 255, 0, 255).astype(np.uint8)
    rgb[~valid] = 0
    alpha = valid.astype(np.uint8) * 255

    profile.update(
        driver="GTiff",
        dtype="uint8",
        count=4,
        nodata=None,
        compress="deflate",
        predictor=2,
        photometric="RGB",
        BIGTIFF="IF_SAFER",
    )

    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(rgb[:, :, 0], 1)
        dst.write(rgb[:, :, 1], 2)
        dst.write(rgb[:, :, 2], 3)
        dst.write(alpha, 4)
        dst.colorinterp = (
            ColorInterp.red,
            ColorInterp.green,
            ColorInterp.blue,
            ColorInterp.alpha,
        )
        dst.write_mask(alpha)

    return output_tif



def find_snaphu(configured: str | None = None):
    candidates = []
    if configured:
        candidates.append(Path(configured))

    found = shutil.which("snaphu") or shutil.which("snaphu.exe")
    if found:
        candidates.append(Path(found))

    search_roots = [
        Path.home() / ".snap" / "auxdata",
        Path.home() / "AppData" / "Roaming" / "SNAP" / "auxdata",
        Path(r"C:\SNAPHU"),
        Path(r"C:\Program Files\SNAPHU"),
    ]

    for root in search_roots:
        if root.exists():
            candidates.extend(root.rglob("snaphu.exe"))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def write_snaphu_export_graph(filtered_dim: Path, snaphu_dir: Path, graph_path: Path):
    from xml.sax.saxutils import escape
    processors = max((os.cpu_count() or 2) - 1, 1)
    xml = f"""<graph id="SnaphuExportGraph">
  <version>1.0</version>
  <node id="Read">
    <operator>Read</operator>
    <sources/>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>{escape(str(filtered_dim))}</file>
    </parameters>
  </node>
  <node id="SnaphuExport">
    <operator>SnaphuExport</operator>
    <sources><sourceProduct refid="Read"/></sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <targetFolder>{escape(str(snaphu_dir))}</targetFolder>
      <statCostMode>DEFO</statCostMode>
      <initMethod>MCF</initMethod>
      <numberOfTileRows>1</numberOfTileRows>
      <numberOfTileCols>1</numberOfTileCols>
      <numberOfProcessors>{processors}</numberOfProcessors>
      <rowOverlap>200</rowOverlap>
      <colOverlap>200</colOverlap>
      <tileCostThreshold>500</tileCostThreshold>
    </parameters>
  </node>
</graph>"""
    graph_path.write_text(xml, encoding="utf-8")


def read_snaphu_parameter(conf_text: str, name: str):
    m = re.search(rf"(?mi)^\s*{re.escape(name)}\s+(.+?)\s*(?:#.*)?$", conf_text)
    if not m:
        return None
    return m.group(1).strip().strip('"').strip("'")


def run_snaphu_unwrap(snaphu_exe: Path, snaphu_conf: Path):
    conf_text = snaphu_conf.read_text(encoding="utf-8", errors="ignore")
    in_name = read_snaphu_parameter(conf_text, "INFILE")
    if in_name:
        wrapped = snaphu_conf.parent / in_name
    else:
        candidates = list(snaphu_conf.parent.glob("Phase*.img"))
        if len(candidates) != 1:
            raise RuntimeError(f"No se pudo identificar la fase envuelta: {candidates}")
        wrapped = candidates[0]

    line_length = read_snaphu_parameter(conf_text, "LINELENGTH")
    if line_length:
        m = re.match(r"\d+", line_length)
        line_length = int(m.group()) if m else None
    else:
        line_length = None

    if line_length is None:
        hdr = wrapped.with_suffix(".hdr")
        txt = hdr.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"(?mi)^\s*samples\s*=\s*(\d+)", txt)
        if not m:
            raise RuntimeError("No se pudo obtener LINELENGTH.")
        line_length = int(m.group(1))

    out_name = read_snaphu_parameter(conf_text, "OUTFILE")
    if out_name:
        unwrapped_img = snaphu_conf.parent / out_name
        unwrapped_hdr = unwrapped_img.with_suffix(".hdr")
    else:
        hdrs = list(snaphu_conf.parent.glob("UnwPhase*.hdr"))
        if len(hdrs) != 1:
            raise RuntimeError(f"No se pudo identificar UnwPhase: {hdrs}")
        unwrapped_hdr = hdrs[0]
        unwrapped_img = unwrapped_hdr.with_suffix(".img")

    cmd = [
        snaphu_exe,
        "-f",
        snaphu_conf.name,
        wrapped.name,
        str(line_length),
    ]

    for _ in run_process(cmd, cwd=snaphu_conf.parent):
        pass

    if not unwrapped_hdr.exists() or not unwrapped_img.exists():
        raise RuntimeError("SNAPHU terminó pero no se encontró la salida desenvuelta.")

    return unwrapped_hdr


def write_post_graph(
    filtered_dim: Path,
    unwrapped_hdr: Path,
    phases_tc: Path,
    los_tc: Path,
    graph_path: Path,
    dem_name: str,
    pixel_spacing: float,
):
    from xml.sax.saxutils import escape

    tc = f"""
      <sourceBands/>
      <demName>{escape(dem_name)}</demName>
      <externalDEMFile/>
      <externalDEMNoDataValue>0.0</externalDEMNoDataValue>
      <externalDEMApplyEGM>true</externalDEMApplyEGM>
      <demResamplingMethod>BILINEAR_INTERPOLATION</demResamplingMethod>
      <imgResamplingMethod>BILINEAR_INTERPOLATION</imgResamplingMethod>
      <pixelSpacingInMeter>{pixel_spacing}</pixelSpacingInMeter>
      <mapProjection>AUTO:42001</mapProjection>
      <saveLocalIncidenceAngle>true</saveLocalIncidenceAngle>
      <saveIncidenceAngleFromEllipsoid>true</saveIncidenceAngleFromEllipsoid>
      <saveSelectedSourceBand>true</saveSelectedSourceBand>
      <nodataValueAtSea>false</nodataValueAtSea>
"""

    xml = f"""<graph id="SnaphuImportAndTerrainCorrection">
  <version>1.0</version>
  <node id="Read-Wrapped">
    <operator>Read</operator><sources/>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>{escape(str(filtered_dim))}</file>
    </parameters>
  </node>
  <node id="Read-Unwrapped">
    <operator>Read</operator><sources/>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>{escape(str(unwrapped_hdr))}</file>
    </parameters>
  </node>
  <node id="SnaphuImport">
    <operator>SnaphuImport</operator>
    <sources>
      <sourceProduct refid="Read-Wrapped"/>
      <sourceProduct.1 refid="Read-Unwrapped"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <doNotKeepWrapped>false</doNotKeepWrapped>
    </parameters>
  </node>
  <node id="Terrain-Correction-Phases">
    <operator>Terrain-Correction</operator>
    <sources><sourceProduct refid="SnaphuImport"/></sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">{tc}</parameters>
  </node>
  <node id="Write-Phases">
    <operator>Write</operator>
    <sources><sourceProduct refid="Terrain-Correction-Phases"/></sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>{escape(str(phases_tc))}</file>
      <formatName>BEAM-DIMAP</formatName>
    </parameters>
  </node>
  <node id="PhaseToDisplacement">
    <operator>PhaseToDisplacement</operator>
    <sources><sourceProduct refid="SnaphuImport"/></sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement"/>
  </node>
  <node id="Terrain-Correction-LOS">
    <operator>Terrain-Correction</operator>
    <sources><sourceProduct refid="PhaseToDisplacement"/></sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">{tc}</parameters>
  </node>
  <node id="Write-LOS">
    <operator>Write</operator>
    <sources><sourceProduct refid="Terrain-Correction-LOS"/></sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>{escape(str(los_tc))}</file>
      <formatName>BEAM-DIMAP</formatName>
    </parameters>
  </node>
</graph>"""
    graph_path.write_text(xml, encoding="utf-8")


def list_img(folder: Path):
    return sorted(folder.glob("*.img"))


def normalized_name(path: Path):
    return re.sub(r"[^a-z0-9]+", "", path.stem.casefold())


def find_img(files, include, exclude=()):
    inc = tuple(re.sub(r"[^a-z0-9]+", "", x.casefold()) for x in include)
    exc = tuple(re.sub(r"[^a-z0-9]+", "", x.casefold()) for x in exclude)
    matches = []
    for f in files:
        n = normalized_name(f)
        if all(x in n for x in inc) and not any(x in n for x in exc):
            matches.append(f)
    return matches[0] if matches else None


def read_raster(path: Path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        mask = src.read_masks(1) > 0
        profile = src.profile.copy()
        grid = (src.width, src.height, src.crs, src.transform)
    return data, mask, profile, grid


def same_grid(a, b):
    return (
        a[0] == b[0]
        and a[1] == b[1]
        and a[2] == b[2]
        and a[3].almost_equals(b[3])
    )


def write_geotiff(path: Path, data, mask, profile, description):
    valid = mask & np.isfinite(data)
    out = np.where(valid, data, -9999.0).astype(np.float32)
    p = profile.copy()
    p.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=-9999.0,
        compress="deflate",
        predictor=3,
        BIGTIFF="IF_SAFER",
    )
    with rasterio.open(path, "w", **p) as dst:
        dst.write(out, 1)
        dst.write_mask(valid.astype(np.uint8) * 255)
        dst.set_band_description(1, description)
    return path


def export_snap_products(phases_tc: Path, los_tc: Path, output_dir: Path, pair_name: str):
    phases_dir = phases_tc.with_suffix(".data")
    los_dir = los_tc.with_suffix(".data")

    phase_files = list_img(phases_dir)
    los_files = list_img(los_dir)

    wrapped_img = find_img(phase_files, ("phase",), ("unwphase",))
    unwrapped_img = find_img(phase_files, ("unwphase",))
    coh_img = find_img(phase_files, ("coh",))
    intensity_img = find_img(phase_files, ("intensity",))
    amplitude_img = find_img(phase_files, ("amplitude",))
    los_img = find_img(los_files, ("displacement",))
    inc_img = find_img(los_files, ("incidenceanglefromellipsoid",))

    required = {
        "wrapped": wrapped_img,
        "unwrapped": unwrapped_img,
        "coherence": coh_img,
        "LOS": los_img,
        "incidence": inc_img,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise RuntimeError("Faltan bandas SNAP: " + ", ".join(missing))

    outputs = {}

    wrapped, m_wrapped, p_wrapped, g_wrapped = read_raster(wrapped_img)
    unwrapped, m_unw, p_unw, g_unw = read_raster(unwrapped_img)
    coh, m_coh, p_coh, g_coh = read_raster(coh_img)
    los, m_los, p_los, g_los = read_raster(los_img)
    inc, m_inc, p_inc, g_inc = read_raster(inc_img)

    if not same_grid(g_wrapped, g_unw) or not same_grid(g_wrapped, g_coh):
        raise RuntimeError("Las bandas de fase/coherencia no comparten grilla.")
    if not same_grid(g_los, g_inc):
        raise RuntimeError("LOS e incidencia no comparten grilla.")

    outputs["wrapped"] = write_geotiff(
        output_dir / f"{pair_name}_wrapped_phase.tif",
        wrapped, m_wrapped, p_wrapped, "wrapped_phase_rad"
    )
    outputs["unwrapped"] = write_geotiff(
        output_dir / f"{pair_name}_unw_phase.tif",
        unwrapped, m_unw, p_unw, "unwrapped_phase_rad"
    )
    outputs["coherence"] = write_geotiff(
        output_dir / f"{pair_name}_corr.tif",
        coh, m_coh, p_coh, "coherence"
    )
    outputs["los"] = write_geotiff(
        output_dir / f"{pair_name}_los_disp.tif",
        los, m_los, p_los, "los_displacement_m"
    )
    outputs["incidence"] = write_geotiff(
        output_dir / f"{pair_name}_incidence_angle.tif",
        inc, m_inc, p_inc, "incidence_angle_from_ellipsoid_deg"
    )

    if amplitude_img is not None:
        amp, m_amp, p_amp, g_amp = read_raster(amplitude_img)
    elif intensity_img is not None:
        intensity, m_amp, p_amp, g_amp = read_raster(intensity_img)
        amp = np.sqrt(np.maximum(intensity, 0.0)).astype(np.float32)
    else:
        raise RuntimeError("No se encontró amplitud/intensidad.")

    if not same_grid(g_wrapped, g_amp):
        raise RuntimeError("La amplitud no comparte grilla con la fase.")

    outputs["amplitude"] = write_geotiff(
        output_dir / f"{pair_name}_amp.tif",
        amp, m_amp, p_amp, "amplitude"
    )

    return outputs



def render_quicklook(tif_path: Path):
    if rasterio is None:
        return None

    with rasterio.open(tif_path) as src:
        if src.count >= 3:
            arr = src.read([1, 2, 3])
            arr = np.moveaxis(arr, 0, -1)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return arr
    return None


init_state()

with st.sidebar:
    st.header("Configuración")

    output_dir = Path(
        st.text_input(
            "Carpeta de salida",
            value=str(Path.home() / "INSAR_Explorer"),
            help="La aplicación guardará aquí descargas, productos HyP3 y salidas SNAP.",
        )
    )

    earthdata_token = st.text_input(
        "Earthdata token",
        type="password",
        help="Se usa para descargar desde ASF y autenticar HyP3.",
    )

    st.divider()
    st.caption("SNAP local")

    gpt_path = Path(
        st.text_input(
            "Ruta gpt.exe",
            value=r"C:\Program Files\esa-snap\bin\gpt.exe",
        )
    )

    default_graph = Path(__file__).resolve().parent / "graphs" / "TOPSAR_Coreg_Interferogram_parametrizado_CORREGIDO.xml"
    snap_graph = Path(
        st.text_input(
            "Grafo SNAP parametrizado",
            value=str(default_graph),
        )
    )

    snaphu_path_text = st.text_input(
        "Ruta snaphu.exe (opcional)",
        value="",
        help="Déjalo vacío para buscarlo automáticamente.",
    )

tabs = st.tabs(
    [
        "1 · Zona y búsqueda",
        "2 · Descargar SLC",
        "3 · HyP3 On Demand",
        "4 · SNAP local",
        "5 · Productos",
    ]
)


with tabs[0]:
    st.subheader("Selecciona el área de estudio")

    left, right = st.columns([2, 1])

    with left:
        m = folium.Map(
            location=[-12.05, -77.03],
            zoom_start=7,
            tiles="CartoDB positron",
            control_scale=True,
        )
        Draw(
            export=False,
            draw_options={
                "polyline": False,
                "circle": False,
                "circlemarker": False,
                "marker": False,
                "polygon": True,
                "rectangle": True,
            },
            edit_options={"edit": True, "remove": True},
        ).add_to(m)
        Fullscreen().add_to(m)
        MeasureControl(position="bottomleft").add_to(m)

        map_state = st_folium(
            m,
            width=None,
            height=520,
            returned_objects=["all_drawings", "last_active_drawing"],
            key="aoi_map",
        )

        geom = extract_latest_polygon(map_state)
        if geom:
            try:
                wkt_value = geometry_to_wkt(geom)
                st.session_state.aoi_geojson = geom
                st.session_state.aoi_wkt = wkt_value
            except Exception as exc:
                st.error(str(exc))

    with right:
        st.write("**AOI actual**")
        if st.session_state.aoi_wkt:
            st.code(st.session_state.aoi_wkt, language=None)
            st.success("AOI listo.")
        else:
            st.info("Dibuja un rectángulo o polígono en el mapa.")

        start_date = st.date_input("Fecha inicial", value=date(2026, 1, 1))
        end_date = st.date_input("Fecha final", value=date.today())
        flight_direction = st.selectbox(
            "Dirección de vuelo",
            ["AMBAS", "ASCENDING", "DESCENDING"],
        )
        max_results = st.number_input(
            "Máximo de escenas",
            min_value=2,
            max_value=500,
            value=100,
            step=10,
        )

        if st.button("Buscar Sentinel-1 SLC", type="primary", use_container_width=True):
            if not st.session_state.aoi_wkt:
                st.error("Primero dibuja un AOI.")
            elif start_date > end_date:
                st.error("La fecha inicial no puede ser posterior a la final.")
            else:
                try:
                    with st.spinner("Consultando catálogo ASF..."):
                        results = search_asf(
                            st.session_state.aoi_wkt,
                            start_date,
                            end_date,
                            flight_direction,
                            max_results,
                        )
                    st.session_state.search_results = results
                    st.success(f"Se encontraron {len(results)} escenas.")
                except Exception as exc:
                    st.exception(exc)

    results = st.session_state.search_results

    if results:
        rows = []
        for i, product in enumerate(results):
            info = safe_props(product)
            info["_index"] = i
            rows.append(info)

        df = pd.DataFrame(rows)

        wanted = [
            "sceneName", "startTime", "flightDirection", "pathNumber",
            "frameNumber", "polarization", "platform", "beamModeType",
        ]
        visible = [c for c in wanted if c in df.columns]
        st.dataframe(df[visible], use_container_width=True, hide_index=True)

        scene_names = [x for x in df["sceneName"].dropna().tolist()]

        selected = st.multiselect(
            "Selecciona escenas para descargar o procesar",
            options=scene_names,
            default=st.session_state.selected_granules,
        )
        st.session_state.selected_granules = selected

        if len(selected) == 2:
            st.success("Par seleccionado.")
        elif len(selected) > 2:
            st.info("Puedes descargar varias escenas. HyP3 y SNAP usarán dos escenas por par.")


with tabs[1]:
    st.subheader("Descargar Sentinel-1 SLC")

    selected = st.session_state.selected_granules
    if not selected:
        st.info("Primero busca y selecciona escenas en la pestaña 1.")
    else:
        st.write(f"Escenas seleccionadas: **{len(selected)}**")
        for name in selected:
            st.code(name, language=None)

        download_dir = output_dir / "SLC"

        if st.button("Descargar desde ASF", type="primary"):
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                download_dir.mkdir(parents=True, exist_ok=True)
                session = make_asf_session(earthdata_token)

                with st.spinner("Descargando..."):
                    found = asf.granule_search(selected)
                    found.download(
                        path=str(download_dir),
                        session=session,
                        processes=min(3, len(selected)),
                    )

                st.success(f"Descarga terminada: {download_dir}")
            except Exception as exc:
                st.exception(exc)

    st.divider()
    st.caption(
        "Para SLC grandes es más robusto usar las rutas locales en la pestaña SNAP. "
        "El navegador no es la mejor vía para subir archivos de varios GB."
    )


with tabs[2]:
    st.subheader("HyP3 On Demand · InSAR GAMMA")

    selected = st.session_state.selected_granules

    g1_default = selected[0] if len(selected) >= 1 else ""
    g2_default = selected[1] if len(selected) >= 2 else ""

    c1, c2 = st.columns(2)

    with c1:
        granule1 = st.text_input("Escena PRE / referencia", value=g1_default)
    with c2:
        granule2 = st.text_input("Escena POST / secundaria", value=g2_default)

    c3, c4, c5 = st.columns(3)

    with c3:
        looks = st.selectbox("Looks", ["20x4", "10x2"], index=1)
    with c4:
        phase_filter = st.slider(
            "Filtro de fase",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
        )
    with c5:
        water_mask = st.checkbox("Máscara de agua", value=True)

    include_wrapped = st.checkbox("Incluir wrapped phase", value=True)
    include_displacement = st.checkbox("Incluir mapas de desplazamiento", value=True)
    job_name = st.text_input("Nombre del trabajo", value="insar_gui_job")

    problems = validate_pair_names(granule1, granule2)
    for problem in problems:
        st.warning(problem)

    if st.button("Enviar a HyP3", type="primary"):
        if sdk is None:
            st.error("Falta instalar hyp3_sdk.")
        elif problems and any("dos escenas" in p for p in problems):
            st.error("Selecciona dos escenas.")
        else:
            try:
                granule1 = normalize_granule_name(granule1)
                granule2 = normalize_granule_name(granule2)

                hyp3 = sdk.HyP3(token=earthdata_token.strip())

                batch = hyp3.submit_insar_job(
                    granule1=granule1,
                    granule2=granule2,
                    name=job_name,
                    looks=looks,
                    phase_filter_parameter=phase_filter,
                    apply_water_mask=water_mask,
                    include_dem=False,
                    include_inc_map=False,
                    include_look_vectors=False,
                    include_displacement_maps=include_displacement,
                    include_wrapped_phase=include_wrapped,
                )

                st.session_state["hyp3_batch"] = batch
                st.success("Trabajo enviado.")
                st.write(batch)
            except Exception as exc:
                st.exception(exc)

    if "hyp3_batch" in st.session_state:
        st.divider()

        if st.button("Actualizar estado HyP3"):
            try:
                hyp3 = sdk.HyP3(token=earthdata_token.strip())
                batch = hyp3.refresh(st.session_state["hyp3_batch"])
                st.session_state["hyp3_batch"] = batch
                st.write(batch)
            except Exception as exc:
                st.exception(exc)

        if st.button("Esperar y descargar producto"):
            try:
                hyp3_dir = output_dir / "HyP3"
                hyp3_dir.mkdir(parents=True, exist_ok=True)

                hyp3 = sdk.HyP3(token=earthdata_token.strip())
                batch = hyp3.watch(
                    st.session_state["hyp3_batch"],
                    timeout=24 * 60 * 60,
                )
                batch.download_files(location=hyp3_dir, create=True)

                st.success(f"Producto descargado en {hyp3_dir}")
            except Exception as exc:
                st.exception(exc)


with tabs[3]:
    st.subheader("Procesamiento local con SNAP")

    st.info(
        "Este modo usa tus SLC .zip locales y el grafo parametrizado de SNAP. "
        "No vuelve a descargar las escenas."
    )

    source_mode = st.radio(
        "Origen de los SLC",
        ["Rutas locales", "Subir archivos ZIP"],
        horizontal=True,
    )

    local_temp_dir = output_dir / "_uploads"

    if source_mode == "Rutas locales":
        master_path = Path(
            st.text_input(
                "SLC master / PRE",
                value=r"D:\INSAR\SLC\PRE.zip",
            )
        )
        secondary_path = Path(
            st.text_input(
                "SLC secondary / POST",
                value=r"D:\INSAR\SLC\POST.zip",
            )
        )
    else:
        uploads = st.file_uploader(
            "Sube exactamente dos Sentinel-1 SLC ZIP",
            type=["zip"],
            accept_multiple_files=True,
        )

        master_path = None
        secondary_path = None

        if uploads and len(uploads) == 2:
            local_temp_dir.mkdir(parents=True, exist_ok=True)
            saved = []
            for uploaded in uploads:
                path = local_temp_dir / uploaded.name
                with open(path, "wb") as f:
                    f.write(uploaded.getbuffer())
                saved.append(path)
            master_path, secondary_path = saved

        elif uploads:
            st.warning("Selecciona exactamente dos ZIP.")

    c1, c2, c3 = st.columns(3)

    with c1:
        subswath = st.selectbox("Subswath", ["IW1", "IW2", "IW3"], index=0)
        first_burst = st.number_input("Primer burst", 1, 20, 1)
        last_burst = st.number_input("Último burst", 1, 20, 4)
    with c2:
        range_looks = st.selectbox("Range looks", [5, 10, 20], index=1)
        azimuth_looks = st.selectbox("Azimuth looks", [1, 2, 4], index=1)
        snap_filter = st.slider("Goldstein", 0.0, 1.0, 0.6, 0.05)
    with c3:
        dem_name = st.selectbox(
            "DEM",
            ["Copernicus 30m Global DEM", "SRTM 3Sec"],
        )
        output_name = st.text_input("Nombre del par", value="S1_PAIR")
        pixel_spacing = st.number_input(
            "Pixel spacing final [m]",
            min_value=10.0,
            max_value=200.0,
            value=80.0,
            step=10.0,
        )

    processing_mode = st.radio(
        "Nivel de procesamiento",
        [
            "Solo interferograma filtrado",
            "Completo: SNAPHU + LOS + GeoTIFF + Color Phase",
        ],
        horizontal=True,
    )

    st.warning(
        "En esta primera versión los bursts se ingresan manualmente. "
        "La detección automática de subswath/bursts desde el AOI será el siguiente módulo."
    )

    if st.button("Ejecutar interferograma SNAP", type="primary"):
        try:
            if master_path is None or secondary_path is None:
                raise ValueError("Faltan los dos SLC.")
            if not master_path.exists():
                raise FileNotFoundError(master_path)
            if not secondary_path.exists():
                raise FileNotFoundError(secondary_path)
            if not gpt_path.exists():
                raise FileNotFoundError(f"No existe gpt.exe: {gpt_path}")
            if not snap_graph.exists():
                raise FileNotFoundError(
                    f"No existe el grafo SNAP: {snap_graph}\n"
                    "Sube o indica TOPSAR_Coreg_Interferogram_parametrizado_CORREGIDO.xml"
                )

            pair_dir = output_dir / "SNAP" / output_name
            pair_dir.mkdir(parents=True, exist_ok=True)
            filtered_dim = pair_dir / f"{output_name}_ifg_ml_flt.dim"

            cmd = [
                gpt_path,
                snap_graph,
                f"-Pmaster={master_path}",
                f"-Pslave={secondary_path}",
                f"-Psubswath={subswath}",
                f"-Pfirst_burst={first_burst}",
                f"-Plast_burst={last_burst}",
                f"-Prange_looks={range_looks}",
                f"-Pazimuth_looks={azimuth_looks}",
                f"-Pphase_filter={snap_filter}",
                f"-Pdem_name={dem_name}",
                "-Porbit_type=Sentinel Precise (Auto Download)",
                f"-Psalida_filtrada={filtered_dim}",
                "-e",
            ]

            log_box = st.empty()
            log = ""
            for line in run_process(cmd):
                log += line
                log_box.code(log[-12000:], language=None)

            st.success(f"Interferograma SNAP generado: {filtered_dim}")
            st.session_state["last_snap_dim"] = str(filtered_dim)

            if processing_mode.startswith("Completo"):
                if rasterio is None:
                    raise RuntimeError("Faltan dependencias rasterio/scipy/matplotlib.")

                snaphu_exe = find_snaphu(snaphu_path_text.strip() or None)
                if snaphu_exe is None:
                    raise FileNotFoundError(
                        "No se encontró snaphu.exe. Indica su ruta en la barra lateral."
                    )

                snaphu_dir = pair_dir / "snaphu"
                snaphu_dir.mkdir(parents=True, exist_ok=True)
                export_graph = pair_dir / "snaphu_export.xml"
                write_snaphu_export_graph(filtered_dim, snaphu_dir, export_graph)

                st.info("Exportando a SNAPHU...")
                for line in run_process([gpt_path, export_graph, "-e"]):
                    log += line
                    log_box.code(log[-12000:], language=None)

                confs = list(snaphu_dir.rglob("snaphu.conf"))
                if len(confs) != 1:
                    raise RuntimeError(f"Se esperaba un snaphu.conf y se encontraron {len(confs)}.")
                snaphu_conf = confs[0]

                st.info("Desenvolviendo fase con SNAPHU...")
                unwrapped_hdr = run_snaphu_unwrap(snaphu_exe, snaphu_conf)

                phases_tc = pair_dir / f"{output_name}_phases_tc.dim"
                los_tc = pair_dir / f"{output_name}_los_disp.dim"
                post_graph = pair_dir / "snaphu_import_tc.xml"
                write_post_graph(
                    filtered_dim,
                    unwrapped_hdr,
                    phases_tc,
                    los_tc,
                    post_graph,
                    dem_name,
                    float(pixel_spacing),
                )

                st.info("Importando SNAPHU y aplicando Terrain Correction...")
                for line in run_process([gpt_path, post_graph, "-e"]):
                    log += line
                    log_box.code(log[-12000:], language=None)

                st.info("Exportando GeoTIFF individuales...")
                outputs = export_snap_products(
                    phases_tc,
                    los_tc,
                    pair_dir,
                    output_name,
                )

                color_out = pair_dir / f"{output_name}_color_phase.tif"
                create_color_phase(
                    outputs["wrapped"],
                    outputs["coherence"],
                    outputs["amplitude"],
                    color_out,
                    sigma=1.0,
                )
                outputs["color_phase"] = color_out

                st.session_state["last_snap_outputs"] = {
                    k: str(v) for k, v in outputs.items()
                }
                st.success("Procesamiento completo terminado.")

        except Exception as exc:
            st.exception(exc)

    st.caption(
        "El grafo corregido ya viene incluido en esta versión. "
        "Puedes ejecutar solo el interferograma o continuar con SNAPHU + Terrain Correction + GeoTIFF."
    )


with tabs[4]:
    st.subheader("Productos y visualización")

    hyp3_root = output_dir / "HyP3"
    snap_root = output_dir / "SNAP"

    st.write("**Carpeta de trabajo**")
    st.code(str(output_dir), language=None)

    if hyp3_root.exists():
        zips = sorted(hyp3_root.glob("*.zip"))
        st.write(f"ZIP HyP3 encontrados: **{len(zips)}**")

        if zips:
            selected_zip = st.selectbox(
                "Producto HyP3",
                zips,
                format_func=lambda p: p.name,
            )

            if st.button("Descomprimir / localizar TIFF HyP3"):
                try:
                    product_dir = extract_hyp3_zip(selected_zip, hyp3_root)
                    st.session_state.last_hyp3_product_dir = str(product_dir)
                    st.success(product_dir)
                except Exception as exc:
                    st.exception(exc)

    product_dir_value = st.session_state.last_hyp3_product_dir

    if product_dir_value:
        product_dir = Path(product_dir_value)

        phase = find_tiff(product_dir, "_wrapped_phase.tif")
        coh = find_tiff(product_dir, "_corr.tif")
        amp = find_tiff(product_dir, "_amp.tif")

        st.write("**Archivos detectados**")
        st.write("Fase:", phase)
        st.write("Coherencia:", coh)
        st.write("Amplitud:", amp)

        if phase and coh and amp:
            out_dir = output_dir / "Productos"
            out_dir.mkdir(parents=True, exist_ok=True)
            color_out = out_dir / f"{product_dir.name}_color_phase.tif"

            if st.button("Generar Color Phase GeoTIFF"):
                try:
                    create_color_phase(
                        phase,
                        coh,
                        amp,
                        color_out,
                        sigma=1.0,
                    )
                    st.success(color_out)
                except Exception as exc:
                    st.exception(exc)

            if color_out.exists():
                arr = render_quicklook(color_out)
                if arr is not None:
                    st.image(arr, caption="Color Phase", use_container_width=True)

                with open(color_out, "rb") as f:
                    st.download_button(
                        "Descargar Color Phase TIFF",
                        data=f.read(),
                        file_name=color_out.name,
                        mime="image/tiff",
                    )

    if snap_root.exists():
        dims = sorted(snap_root.rglob("*_ifg_ml_flt.dim"))
        if dims:
            st.write("**Productos SNAP locales**")
            for dim in dims[-10:]:
                st.code(str(dim), language=None)

    snap_outputs = st.session_state.get("last_snap_outputs")
    if snap_outputs:
        st.divider()
        st.write("**Últimos productos SNAP completos**")
        for key, value in snap_outputs.items():
            st.write(f"{key}:")
            st.code(value, language=None)

        color_path = Path(snap_outputs.get("color_phase", ""))
        if color_path.exists():
            arr = render_quicklook(color_path)
            if arr is not None:
                st.image(arr, caption="Color Phase SNAP", use_container_width=True)
