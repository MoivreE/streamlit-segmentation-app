import streamlit as st
import torch
import rasterio
import rasterio.io
import rasterio.features
import rasterio.crs # Importar CRS explícitamente
import numpy as np
import io # Necesario para leer bytes
import os
import geopandas as gpd
from shapely.geometry import shape
import folium
from streamlit_folium import st_folium
# Asegúrate de que el archivo resuneta.py esté en el mismo directorio o sea instalable
try:
    from resuneta import ResUNetA # La arquitectura base del modelo
except ImportError:
    st.error("Error Crítico: No se pudo importar la clase 'ResUNetA' desde 'resuneta.py'.")
    st.error("Asegúrate de que el archivo 'resuneta.py' exista en el mismo directorio que este script.")
    st.stop() # Detener si no se puede importar el modelo

import zipfile # Para crear el zip del shapefile
import tempfile # Para crear archivos/directorios temporales
import traceback # Para imprimir errores detallados
import math # Para cálculos de tiling

# --- Configuración y Constantes ---
st.set_page_config(page_title="Segmentación Satelital (Tiling)", layout="wide")

# --- Parámetros de Tiling ---
TILE_SIZE = 256
OVERLAP = 32

# Determinar ruta al modelo POR DEFECTO
try:
    # Obtener el directorio del script actual
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    # Construir la ruta completa al modelo dentro de la carpeta 'models'
    DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "best_model.pth")
except NameError:
    # __file__ no está definido si se ejecuta interactivamente
    st.warning("No se pudo determinar la ruta del script automáticamente. "
               "Asegúrate de que 'models/best_model.pth' sea accesible "
               "desde el directorio donde ejecutas 'streamlit run'.")
    DEFAULT_MODEL_PATH = "models/best_model.pth" # Usar ruta relativa como fallback

# Configuración del dispositivo (GPU si está disponible, sino CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parámetros de normalización (asegurarse de que sean arrays NumPy)
mean = np.array([159.42794111288646, 227.57129377067616, 248.58301866956222, 704.0244284536792])
std = np.array([304.33628952428364, 422.21148080754443, 508.2267116736817, 1282.962544017454])

# Umbral para convertir probabilidades en máscara binaria (0 o 1)
threshold_default = 0.7

# --- Funciones de Carga de Modelo ---

@st.cache_resource # Cachear el modelo por defecto basado en su ruta
def load_default_model(model_path):
    """Carga el modelo PyTorch por defecto desde la ruta especificada."""
    st.info(f"Cargando modelo por defecto desde: {model_path}")
    if not os.path.exists(model_path):
         st.error(f"Error Crítico: No se encontró el modelo por defecto en '{model_path}'.")
         return None
    try:
        # Asume que el modelo por defecto también usa la arquitectura ResUNetA
        model = ResUNetA().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # Poner el modelo en modo evaluación
        print(f"[INFO] Modelo por defecto cargado exitosamente.")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo por defecto: {e}")
        st.code(traceback.format_exc())
        return None

@st.cache_resource # Cachear modelo subido basado en sus bytes
def load_uploaded_model(model_bytes):
    """Carga un modelo PyTorch (arquitectura ResUNetA) desde bytes."""
    st.info("Intentando cargar el modelo subido...")
    try:
        # --- ¡¡ASUNCIÓN IMPORTANTE!! ---
        # Se asume que el modelo subido tiene la MISMA arquitectura (ResUNetA)
        model = ResUNetA().to(device)
        # Crear un buffer de bytes en memoria para torch.load
        buffer = io.BytesIO(model_bytes)
        # Cargar el state_dict desde el buffer
        model.load_state_dict(torch.load(buffer, map_location=device))
        model.eval()
        print("[INFO] Modelo subido cargado exitosamente.")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo subido: {e}")
        st.error("Posibles causas: El archivo no es un '.pth' válido, "
                 "o la arquitectura del modelo guardado no coincide con 'ResUNetA'.")
        st.code(traceback.format_exc())
        return None

# --- Funciones Auxiliares (Predicción, Procesamiento Geoespacial) ---
# (normalize_image, denormalize_image, predict_tile_probabilities,
#  predict_large_image_tiled, create_mask_tif_bytes,
#  create_shapefile_zip_bytes - sin cambios respecto a la versión anterior)

def normalize_image(image, mean_vals, std_vals):
    mean_vals = mean_vals[:, None, None]; std_vals = std_vals[:, None, None]
    std_vals[std_vals < 1e-6] = 1e-6
    return (image - mean_vals) / std_vals

def denormalize_image(image, mean_vals, std_vals):
    mean_vals = mean_vals[:, None, None].astype(np.float32); std_vals = std_vals[:, None, None].astype(np.float32)
    image_rgb = image.astype(np.float32)
    image_rgb = image_rgb * std_vals + mean_vals; image_rgb = np.transpose(image_rgb, (1, 2, 0))
    min_val, max_val = np.min(image_rgb), np.max(image_rgb)
    print(f"[DEBUG] Denorm range: Min={min_val:.2f}, Max={max_val:.2f}")
    if max_val > min_val: image_display = ((image_rgb - min_val) / (max_val - min_val + 1e-8)) * 255.0
    else: image_display = np.full_like(image_rgb, 128)
    image_display = np.clip(image_display, 0, 255).astype(np.uint8)
    print(f"[DEBUG] Display range: Min={image_display.min()}, Max={image_display.max()}")
    return image_display

def predict_tile_probabilities(tile_np, model, device, mean_vals, std_vals):
    tile_normalized = normalize_image(tile_np.astype(np.float32), mean_vals, std_vals)
    tile_tensor = torch.from_numpy(tile_normalized).unsqueeze(0).to(device).float()
    with torch.no_grad(): output = model(tile_tensor); pred_prob = torch.sigmoid(output)
    pred_prob_np = pred_prob.cpu().numpy().squeeze()
    return pred_prob_np

def predict_large_image_tiled(src, model, device, current_threshold, mean_vals, std_vals, tile_size=TILE_SIZE, overlap=OVERLAP):
    height = src.height; width = src.width; num_bands = src.count
    print(f"[INFO] Tiling {width}x{height}... Tile:{tile_size}, Overlap:{overlap}")
    sum_probs = np.zeros((height, width), dtype=np.float32); counts = np.zeros((height, width), dtype=np.uint8)
    step = tile_size - overlap; step = max(1, step) # Ensure step is at least 1
    n_tiles_x = math.ceil(width / step); n_tiles_y = math.ceil(height / step); total_tiles = n_tiles_x * n_tiles_y
    processed_tiles = 0; progress_bar = st.progress(0.0, text="Iniciando tiles...")
    for r in range(0, height, step):
        for c in range(0, width, step):
            processed_tiles += 1; progress_value = float(processed_tiles) / total_tiles
            progress_bar.progress(progress_value, text=f"Tile {processed_tiles}/{total_tiles}")
            row_start, col_start = r, c; current_tile_h = min(tile_size, height - row_start); current_tile_w = min(tile_size, width - col_start)
            window = rasterio.windows.Window(col_start, row_start, current_tile_w, current_tile_h)
            print(f"[DEBUG] Tile ({processed_tiles}/{total_tiles}): Origin=({r},{c}), WinSize=({current_tile_w},{current_tile_h})")
            tile_data = src.read(window=window)
            pad_h_bottom = tile_size - current_tile_h; pad_w_right = tile_size - current_tile_w
            if pad_h_bottom > 0 or pad_w_right > 0:
                print(f"[DEBUG] Padding: H={pad_h_bottom}, W={pad_w_right}")
                tile_data = np.pad(tile_data, ((0, 0), (0, pad_h_bottom), (0, pad_w_right)), mode='reflect')
            tile_probs = predict_tile_probabilities(tile_data, model, device, mean_vals, std_vals)
            row_end = row_start + current_tile_h; col_end = col_start + current_tile_w
            sum_probs[row_start:row_end, col_start:col_end] += tile_probs[:current_tile_h, :current_tile_w]
            counts[row_start:row_end, col_start:col_end] += 1
    progress_bar.progress(1.0, text="Ensamblaje final..."); counts[counts == 0] = 1; avg_probs = sum_probs / counts
    print(f"[DEBUG] Avg prob range: Min={np.min(avg_probs):.4f}, Max={np.max(avg_probs):.4f}")
    final_mask = (avg_probs > current_threshold).astype(np.uint8)
    print(f"[INFO] Máscara final ({width}x{height}) creada (umbral={current_threshold:.2f})")
    return final_mask

def create_mask_tif_bytes(mask, profile):
    profile.update(dtype=rasterio.uint8, count=1, nodata=None)
    memfile = io.BytesIO();
    with rasterio.io.MemoryFile(memfile) as mem_dst:
        with mem_dst.open(**profile) as dst: dst.write(mask, 1)
    memfile.seek(0); return memfile.read()

def create_shapefile_zip_bytes(mask, transform, crs):
    gdf = None;
    try: shapes_gen = rasterio.features.shapes(mask.astype(rasterio.uint8), mask=(mask == 1), transform=transform); geometries = [shape(geom) for geom, value in shapes_gen if value == 1]
    except Exception as e: print(f"[ERROR] Shapes: {e}"); print(traceback.format_exc()); return None, None
    if not geometries: print("[INFO] No geometries found."); return None, None
    try: gdf = gpd.GeoDataFrame(geometry=geometries, crs=crs); gdf['class_id'] = 1; print(f"[INFO] GDF created: {len(gdf)} polygons.")
    except Exception as e: print(f"[ERROR] GDF creation: {e}"); print(traceback.format_exc()); return None, None
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_shp = os.path.join(temp_dir, "temp.shp"); print(f"[DEBUG] Saving temp SHP: {temp_shp}")
            gdf.to_file(temp_shp, driver='ESRI Shapefile', encoding='utf-8'); print(f"[DEBUG] Temp SHP saved.")
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                print("[DEBUG] Zipping files...")
                files_to_add = {"predicted_mask.shp": temp_shp.replace(".shp", ".shp"),
                                "predicted_mask.shx": temp_shp.replace(".shp", ".shx"),
                                "predicted_mask.dbf": temp_shp.replace(".shp", ".dbf"),
                                "predicted_mask.prj": temp_shp.replace(".shp", ".prj")}
                for arcname, filepath in files_to_add.items():
                    if os.path.exists(filepath): print(f"[DEBUG] Adding {arcname}..."); zf.write(filepath, arcname=arcname)
                    elif arcname != "predicted_mask.prj": print(f"[WARN] Missing {filepath}")
                    elif gdf.crs: print(f"[WARN] Missing {filepath} (.prj)")
                    else: print(f"[DEBUG] Missing {filepath} (.prj), esperado (sin CRS)")
            zip_buf.seek(0); print("[INFO] ZIP created."); return zip_buf.read(), gdf
    except Exception as e: print(f"[ERROR] Tempfile/Zip: {e}"); print(traceback.format_exc()); return None, gdf

# --- Interfaz de Streamlit ---
st.title("Segmentación Satelital con Tiling (Modelo Flexible)")
st.write("""
    Carga una imagen **GeoTIFF de 4 bandas**. Puedes usar el modelo por defecto
    o subir tu propio modelo `.pth` (debe tener la arquitectura `ResUNetA`).
    La aplicación procesará por tiles, predecirá, ensamblará la máscara,
    generará Shapefile y mostrará el mapa.
""")

# --- Barra Lateral (Sidebar) ---
st.sidebar.title("Configuración")
st.sidebar.info("Modelo Base: ResUNet-a")
st.sidebar.markdown("---")

# ---- 1. Selección de Modelo ----
st.sidebar.subheader("1. Selección de Modelo")
model_source = st.sidebar.radio(
    "Elige la fuente del modelo:",
    ("Usar Modelo por Defecto (Guárico)", "Subir mi Modelo (.pth)"),
    key="model_source_radio" # Clave única
)

model_to_use = None # Variable para almacenar el modelo que se usará
uploaded_model_file = None # Para el archivo subido

if model_source == "Usar Modelo por Defecto (Guárico)":
    model_to_use = load_default_model(DEFAULT_MODEL_PATH)
    if model_to_use is None: st.sidebar.error("Fallo al cargar el modelo por defecto.")
else:
    uploaded_model_file = st.sidebar.file_uploader(
        "Carga tu archivo .pth", type=["pth"], key="model_uploader",
        help="El modelo DEBE ser compatible con la arquitectura ResUNetA."
    )
    if uploaded_model_file is not None:
        model_bytes = uploaded_model_file.getvalue()
        # Usar cache basado en bytes del archivo subido
        model_to_use = load_uploaded_model(model_bytes)
        if model_to_use is None: st.sidebar.error("Fallo al cargar el modelo subido.")
    else:
        st.sidebar.info("Esperando archivo '.pth'...")

# Mensaje sobre el modelo activo
if model_to_use:
    if model_source == "Usar Modelo por Defecto (Guárico)": st.sidebar.success("✔️ Usando modelo por defecto.")
    elif uploaded_model_file: st.sidebar.success(f"✔️ Usando modelo: {uploaded_model_file.name}")
else:
    st.sidebar.warning("⚠️ Esperando selección/carga de modelo.")
    # No detener aquí, esperar a que se cargue imagen también

# ---- 2. Parámetros de Predicción ----
st.sidebar.markdown("---")
st.sidebar.subheader("2. Parámetros de Predicción")
threshold_slider = st.sidebar.slider(
    "Umbral de Predicción", 0.1, 0.9, threshold_default, 0.05, key="threshold_slider",
    help="Probabilidad mínima (0-1) para clasificar píxel como cultivo."
)
threshold_current = threshold_slider

st.sidebar.markdown("---")
st.sidebar.info(f"Dispositivo: **{str(device).upper()}**")

st.sidebar.info(    """
    Esta aplicación utiliza un modelo **ResUNet-a** para segmentar áreas que podrían
    corresponder a cultivos de arroz en imágenes satelitales **GeoTIFF**.

    **Requisitos de Entrada:**
    * Formato: GeoTIFF (.tif, .tiff)
    * Bandas: 4

    **Pasos:**
    1.  Carga tu imagen usando el botón de arriba.
    2.  Observa la imagen original (RGB aprox.) y la máscara binaria predicha.
    3.  Descarga los resultados si lo deseas:
        * **Máscara (.tif):** La máscara binaria como GeoTIFF.
        * **Shapefile (.zip):** Los polígonos de las áreas predichas.
    4.  Explora los polígonos en el mapa interactivo.
    """)

# --- Sección Principal ---
# Widget para cargar imagen
uploaded_image_file = st.file_uploader(
    "Selecciona una imagen GeoTIFF (.tif, .tiff)",
    type=["tif", "tiff"],
    key="imageUploader"
)

# >> INICIO Bloque Principal de Procesamiento <<
# Procesar solo si se carga una imagen *Y* hay un modelo cargado y listo
if uploaded_image_file is not None and model_to_use is not None:
    print(f"\n[INFO] Procesando imagen: {uploaded_image_file.name}")
    gdf = None # Resetear gdf para nueva imagen
    try:
        bytes_data = uploaded_image_file.getvalue()
        with rasterio.open(io.BytesIO(bytes_data)) as src:
            # ---- 1. Leer Metadatos ----
            profile = src.profile; transform = src.transform; crs = src.crs
            height = src.height; width = src.width; count = src.count
            try: dtype_raster = src.dtypes[0]
            except IndexError: st.error("Error leyendo dtypes."); st.stop()
            print(f"[INFO] Metadatos: {height}x{width} px, {count} bandas, {dtype_raster}, CRS: {crs}")

            # ---- 2. Validar Bandas ----
            if count != 4: st.error(f"Error: Se esperan 4 bandas, detectadas: {count}."); st.stop()

            # ---- 3. Manejar CRS ----
            original_crs = crs
            if crs is None:
                st.warning("Imagen sin CRS. Asumiendo EPSG:4326."); crs = rasterio.crs.CRS.from_epsg(4326)
                profile['crs'] = crs; profile['transform'] = transform
            else: st.info(f"CRS detectado: {crs.to_string()}")

            # ---- 4. Predicción por Tiles (Pasando el modelo seleccionado) ----
            st.subheader("1. Predicción por Tiles")
            with st.spinner(f"Procesando {width}x{height} por tiles..."):
                final_large_mask = predict_large_image_tiled(
                    src, model_to_use, device, threshold_current, mean, std, TILE_SIZE, OVERLAP
                )

            # ---- 5. Visualización ----
            st.subheader("2. Resultados Visuales (Miniaturas/Info)")
            col1, col2 = st.columns(2)
            max_display_dim = 1024
            with col1: # Imagen Original
                st.write(f"Original ({width}x{height})")
                if width > max_display_dim or height > max_display_dim: st.info(f"Imagen muy grande para mostrar.")
                else:
                    try:
                        band_indices_rgb = [2, 1, 0]; # AJUSTA ORDEN RGB
                        if max(band_indices_rgb)>=count: st.error(f"Índices RGB inválidos ({count} bandas).")
                        else:
                            image_np_full = src.read().astype(np.float32)
                            image_display = denormalize_image(image_np_full[band_indices_rgb], mean[band_indices_rgb], std[band_indices_rgb])
                            st.image(image_display, caption=f"Bandas {band_indices_rgb[0]+1},{band_indices_rgb[1]+1},{band_indices_rgb[2]+1}", use_container_width=True)
                    except Exception as e: st.error(f"Error mostrando original: {e}"); print(f"[ERROR] Vis Orig: {traceback.format_exc()}")
            with col2: # Máscara Predicha
                st.write(f"Máscara ({width}x{height})")
                if final_large_mask is not None:
                     if np.any(final_large_mask):
                         if width > max_display_dim or height > max_display_dim: st.info(f"Máscara muy grande para mostrar.")
                         else: st.image((final_large_mask * 255).astype(np.uint8), caption="Máscara Binaria", use_container_width=True)
                     else: st.info(f"Máscara vacía (umbral={threshold_current:.2f})."); st.image(np.zeros((min(height, max_display_dim//2), min(width,max_display_dim//2)), dtype=np.uint8), caption="Vacía", use_container_width=True)
                else: st.warning("Máscara no generada.")

            # ---- 6. Descargas ----
            st.subheader("3. Descargas")
            if final_large_mask is not None:
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1: # TIF
                    st.write("**Máscara GeoTIFF:**")
                    try:
                        mask_profile = profile.copy(); mask_profile.update(dtype=rasterio.uint8, count=1, nodata=None, height=final_large_mask.shape[0], width=final_large_mask.shape[1], transform=transform, crs=crs)
                        tif_bytes = create_mask_tif_bytes(final_large_mask, mask_profile)
                        tif_filename = f"mask_{os.path.splitext(uploaded_image_file.name)[0]}.tif"; st.download_button("⬇️ .tif", tif_bytes, tif_filename, "image/tiff")
                    except Exception as e: st.error(f"Error TIF: {e}"); print(f"[ERROR] TIF: {traceback.format_exc()}")
                with col_dl2: # Shapefile ZIP
                    st.write("**Shapefile:**")
                    try:
                        zip_bytes, gdf = create_shapefile_zip_bytes(final_large_mask, transform, crs)
                        if zip_bytes and gdf is not None and not gdf.empty:
                             zip_filename = f"shapefile_{os.path.splitext(uploaded_image_file.name)[0]}.zip"; st.download_button("⬇️ .zip", zip_bytes, zip_filename, "application/zip")
                        elif gdf is None or gdf.empty: st.info("No hay polígonos.")
                        else: st.error("Error generando ZIP.")
                    except Exception as e: st.error(f"Error ZIP: {e}"); print(f"[ERROR] ZIP: {traceback.format_exc()}")
            else: st.warning("Sin máscara, descargas no disponibles.")

            # ---- 7. Mapa Interactivo ----
            st.subheader("4. Mapa Interactivo")
            if gdf is not None and not gdf.empty:
                with st.spinner("🗺️ Generando mapa..."):
                    try:
                        # --- Bloque CRS Mapa CORREGIDO ---
                        gdf_display = None # Inicializar

                        if gdf.crs:
                            # El GeoDataFrame TIENE un CRS definido
                            try:
                                # Comparar con EPSG:4326 (WGS84)
                                if gdf.crs != "EPSG:4326":
                                    print(f"[INFO] Reproyectando GDF de {gdf.crs.to_string()} a EPSG:4326 para mapa.")
                                    gdf_display = gdf.to_crs("EPSG:4326")
                                else:
                                    print("[INFO] GDF ya está en EPSG:4326.")
                                    gdf_display = gdf
                            except Exception as e_crs_comp_repr:
                                st.error(f"Error al procesar/reproyectar CRS {gdf.crs.to_string()}: {e_crs_comp_repr}")
                                st.warning("No se pudo preparar GDF para el mapa.")
                                # gdf_display se mantiene como None
                        else:
                            # El GeoDataFrame NO TIENE un CRS definido
                            st.warning("GDF sin CRS definido. Asumiendo EPSG:4326 para el mapa. "
                                       "La ubicación puede ser incorrecta.")
                            # --- CORRECCIÓN DE SINTAXIS APLICADA AQUÍ ---
                            try:
                                gdf.crs = "EPSG:4326" # Intentar asignar CRS
                                gdf_display = gdf    # Si tiene éxito, usar este gdf
                            # --- FIN CORRECCIÓN DE SINTAXIS ---
                            except Exception as e_crs_assign:
                                st.error(f"No se pudo asignar EPSG:4326 al GDF sin CRS: {e_crs_assign}")
                                # gdf_display se mantiene como None (ya inicializado antes del if/else)

                        # --- Fin Bloque CRS Mapa ---

                        # Continuar solo si gdf_display se pudo preparar
                        if gdf_display is not None:
                            bounds = gdf_display.total_bounds
                            if np.isfinite(bounds).all() and (bounds[2] > bounds[0]) and (bounds[3] > bounds[1]):
                                center_y, center_x = (bounds[1] + bounds[3]) / 2.0, (bounds[0] + bounds[2]) / 2.0
                                m = folium.Map(location=[center_y, center_x], zoom_start=12, tiles="CartoDB positron")
                                folium.GeoJson(gdf_display, name="Predicción", style_function=lambda x: {"fillColor":"#2ca02c", "color":"#006400", "weight":1, "fillOpacity":0.6}, tooltip=folium.features.GeoJsonTooltip(fields=['class_id'], aliases=['ID:'])).add_to(m)
                                folium.LayerControl().add_to(m); st_folium(m, width='100%', height=600, returned_objects=[])
                            else: st.warning("Límites inválidos en GDF.")
                        else: st.warning("No se pudo preparar GDF para mapa.")
                    except Exception as e_map: st.error(f"Error mapa: {e_map}"); st.code(traceback.format_exc())
            elif gdf is None or gdf.empty: st.info("Mapa no disponible (sin polígonos).")

    # ---- Manejo de Errores Generales ----
    except rasterio.RasterioIOError as e: st.error(f"Error leyendo GeoTIFF: {e}"); print(f"[ERROR] Rasterio: {traceback.format_exc()}")
    except ImportError as e: st.error(f"Error importación: {e}"); print(f"[ERROR] Import: {traceback.format_exc()}")
    except Exception as e: st.error(f"Error inesperado: {e}"); print(f"[ERROR] General: {traceback.format_exc()}")

# Mensaje inicial o si falta algo
elif model_to_use is None and uploaded_image_file is None:
     st.info("Selecciona una fuente de modelo y carga una imagen GeoTIFF.")
elif model_to_use is not None and uploaded_image_file is None:
    st.info("Modelo cargado. Esperando a que se cargue una imagen GeoTIFF...")
# El caso de no modelo pero sí imagen ya se maneja con el st.stop() en la sidebar
