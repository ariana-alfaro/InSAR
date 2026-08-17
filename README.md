# InSAR Explorer MVP

Aplicación local en Python/Streamlit para:

- dibujar un AOI en un mapa Leaflet/Folium;
- buscar Sentinel-1 IW SLC en ASF;
- seleccionar y descargar escenas;
- enviar pares a HyP3 On Demand (GAMMA);
- usar SLC `.zip` locales sin volver a descargarlos;
- ejecutar el grafo local de SNAP mediante `gpt.exe`;
- localizar productos HyP3 y generar un `Color Phase` GeoTIFF.

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run insar_explorer.py
```

## SNAP

La aplicación espera por defecto:

```text
C:\Program Files\esa-snap\bin\gpt.exe
```

y un grafo parametrizado compatible con estos parámetros:

- `master`
- `slave`
- `subswath`
- `first_burst`
- `last_burst`
- `range_looks`
- `azimuth_looks`
- `phase_filter`
- `dem_name`
- `orbit_type`
- `salida_filtrada`

Para completar el flujo local igual al notebook corregido todavía hay que incorporar el XML:

`TOPSAR_Coreg_Interferogram_parametrizado_CORREGIDO.xml`

La siguiente versión puede integrar también automáticamente:

1. detección de subswath y bursts desde el AOI;
2. SNAPHU Export;
3. ejecución de SNAPHU;
4. SNAPHU Import;
5. Terrain Correction;
6. fase envuelta GeoTIFF;
7. fase desenvuelta GeoTIFF;
8. coherencia GeoTIFF;
9. desplazamiento LOS GeoTIFF;
10. Color Phase y PNG de presentación.

## Archivos Sentinel-1 grandes

Para SLC de varios GB se recomienda usar **Rutas locales** en vez de subirlos por el navegador.


## Versión 2

Esta versión incluye directamente el grafo:

`graphs/TOPSAR_Coreg_Interferogram_parametrizado_CORREGIDO.xml`

y añade dos modos para SNAP local:

- Solo interferograma filtrado.
- Flujo completo: interferograma -> SNAPHU Export -> SNAPHU -> SNAPHU Import -> Terrain Correction -> GeoTIFF -> Color Phase.

Todavía se seleccionan manualmente `IW1/IW2/IW3` y el rango de bursts. La automatización de bursts desde el AOI queda como siguiente mejora.
