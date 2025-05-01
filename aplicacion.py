# -*- coding: utf-8 -*-
"""
Aplicación Web Streamlit para Segmentación Semántica de Cultivos de Arroz
(Versión Final v8: Tiling, Modelo Flexible, Miniaturas, Fix TIF, Modos, Mapas Base Múltiples con Atribución)
"""

# --- Importaciones Principales ---
import streamlit as st
import torch
import rasterio, rasterio.io, rasterio.features, rasterio.windows, rasterio.crs, rasterio.enums
import numpy as np
import io, os, zipfile, tempfile, traceback, math
import geopandas as gpd
from shapely.geometry import shape
import folium # Importación principal de Folium
from streamlit_folium import st_folium
import pandas as pd
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try: from resuneta import ResUNetA
except ImportError: st.error("Error Crítico: No se pudo importar 'ResUNetA'. Verifica 'resuneta.py'."); st.stop()

# --- Configuración y Constantes ---
st.set_page_config(page_title="Segmentación Arroz (Único/Lote)", layout="wide", initial_sidebar_state="expanded")
TILE_SIZE = 256; OVERLAP = 32
try: SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__)); DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "best_model.pth")
except NameError: st.warning("Ruta script no detectada. Usando 'models/best_model.pth'."); DEFAULT_MODEL_PATH = "models/best_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mean = np.array([159.42794111288646, 227.57129377067616, 248.58301866956222, 704.0244284536792])
std = np.array([304.33628952428364, 422.21148080754443, 508.2267116736817, 1282.962544017454])
threshold_default = 0.7
MAX_DISPLAY_DIM = 1024
MAX_DISPLAY_DIM_PREVIEW = 512

# --- Definición de Mapas Base ---
# Guardamos tanto el identificador/objeto como la atribución requerida
BASEMAPS = {
    "Claro (CartoDB Positron)": {
        "tile": "CartoDB positron",
        "attr": '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
    },
    "Calles (OpenStreetMap)": {
        "tile": "OpenStreetMap",
        "attr": '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    },
    "Satélite (Esri World Imagery)": {
        "tile": folium.TileLayer( # Guardamos el objeto ya que no es un nombre simple
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri', name='Satélite (Esri)', overlay=False, control=True
        ),
        "attr": "Esri" # La atribución ya está en el objeto, pero la guardamos por si acaso
    }
}
BASEMAP_OPTIONS = list(BASEMAPS.keys())

# --- Funciones de Carga de Modelo ---
@st.cache_resource
def load_default_model(model_path):
    print(f"[INFO] Intentando cargar modelo por defecto desde: {model_path}")
    if not os.path.exists(model_path): st.error(f"Error Crítico: No se encontró modelo por defecto en '{model_path}'."); return None
    try: model = ResUNetA().to(device); model.load_state_dict(torch.load(model_path, map_location=device)); model.eval(); print(f"[INFO] Modelo por defecto cargado."); return model
    except Exception as e: st.error(f"Error cargando modelo por defecto: {e}"); st.code(traceback.format_exc()); return None
@st.cache_resource
def load_uploaded_model(model_bytes):
    print("[INFO] Intentando cargar modelo subido...")
    try: model = ResUNetA().to(device); buffer = io.BytesIO(model_bytes); model.load_state_dict(torch.load(buffer, map_location=device)); model.eval(); print("[INFO] Modelo subido cargado."); return model
    except Exception as e: st.error(f"Error cargando modelo subido: {e}"); st.error("Causas posibles: Archivo inválido o arquitectura incompatible."); st.code(traceback.format_exc()); return None

# --- Funciones Auxiliares de Procesamiento ---
def normalize_image(image, mean_vals, std_vals):
    mean_vals=mean_vals[:,None,None];std_vals=std_vals[:,None,None];std_vals[std_vals<1e-6]=1e-6;return (image-mean_vals)/std_vals
def denormalize_image(image, mean_vals, std_vals):
    mean_vals=mean_vals[:,None,None].astype(np.float32);std_vals=std_vals[:,None,None].astype(np.float32);image_rgb=image.astype(np.float32)
    image_rgb=image_rgb*std_vals+mean_vals;image_rgb=np.transpose(image_rgb,(1,2,0));min_val,max_val=np.min(image_rgb),np.max(image_rgb)
    print(f"[DEBUG] Denorm range: Min={min_val:.2f}, Max={max_val:.2f}")
    if max_val>min_val: image_display=((image_rgb-min_val)/(max_val-min_val+1e-8))*255.0
    else: image_display=np.full_like(image_rgb,128)
    image_display=np.clip(image_display,0,255).astype(np.uint8); print(f"[DEBUG] Display range: Min={image_display.min()}, Max={image_display.max()}"); return image_display
def predict_tile_probabilities(tile_np, model, device, mean_vals, std_vals):
    tile_normalized=normalize_image(tile_np.astype(np.float32),mean_vals,std_vals);tile_tensor=torch.from_numpy(tile_normalized).unsqueeze(0).to(device).float()
    with torch.no_grad(): output=model(tile_tensor); pred_prob=torch.sigmoid(output)
    pred_prob_np=pred_prob.cpu().numpy().squeeze(); return pred_prob_np
def predict_large_image_tiled(src, model, device, current_threshold, mean_vals, std_vals,
                              tile_size=TILE_SIZE, overlap=OVERLAP,
                              progress_bar=None, total_tiles_overall=0, processed_offset=0):
    height=src.height;width=src.width;num_bands=src.count;print(f"[INFO] Tiling {width}x{height}... Tile:{tile_size}, Overlap:{overlap}")
    sum_probs=np.zeros((height,width),dtype=np.float32);counts=np.zeros((height,width),dtype=np.uint8);step=tile_size-overlap;step=max(1,step)
    n_tiles_x_local=math.ceil(width/step);n_tiles_y_local=math.ceil(height/step);total_tiles_local=max(1, n_tiles_x_local*n_tiles_y_local)
    processed_tiles_local=0
    for r in range(0,height,step):
        for c in range(0,width,step):
            processed_tiles_local+=1
            if progress_bar and total_tiles_overall > 0:
                 current_overall_tile_count=processed_offset+processed_tiles_local
                 overall_progress_value=float(current_overall_tile_count)/total_tiles_overall
                 progress_text = f"Tile {processed_tiles_local}/{total_tiles_local} (Total: {current_overall_tile_count}/{total_tiles_overall})"
                 progress_bar.progress(min(1.0, overall_progress_value), text=progress_text)
            row_start,col_start=r,c;current_tile_h=min(tile_size,height-row_start);current_tile_w=min(tile_size,width-col_start)
            window=rasterio.windows.Window(col_start,row_start,current_tile_w,current_tile_h);tile_data=src.read(window=window)
            pad_h_bottom=tile_size-current_tile_h;pad_w_right=tile_size-current_tile_w
            if pad_h_bottom>0 or pad_w_right>0: tile_data=np.pad(tile_data,((0,0),(0,pad_h_bottom),(0,pad_w_right)),mode='reflect')
            tile_probs=predict_tile_probabilities(tile_data,model,device,mean_vals,std_vals);row_end=row_start+current_tile_h;col_end=col_start+current_tile_w
            sum_probs[row_start:row_end,col_start:col_end]+=tile_probs[:current_tile_h,:current_tile_w];counts[row_start:row_end,col_start:col_end]+=1
    counts[counts==0]=1;avg_probs=sum_probs/counts;print(f"[DEBUG] Avg prob range: Min={np.min(avg_probs):.4f}, Max={np.max(avg_probs):.4f}")
    final_mask=(avg_probs>current_threshold).astype(np.uint8);print(f"[INFO] Máscara final ({width}x{height}) creada (umbral={current_threshold:.2f})");
    return final_mask, total_tiles_local
def create_mask_tif_bytes(mask, profile):
    temp_tif_path=None;tif_bytes=None
    try:
        with tempfile.NamedTemporaryFile(suffix=".tif",delete=False) as temp_f:temp_tif_path=temp_f.name
        print(f"[DEBUG] Creando TIF temporal: {temp_tif_path}");profile.update(driver='GTiff',dtype=rasterio.uint8,count=1)
        with rasterio.open(temp_tif_path,'w',**profile) as dst:dst.write(mask.astype(rasterio.uint8),1)
        print(f"[DEBUG] TIF temporal escrito.");
        with open(temp_tif_path,'rb') as f:tif_bytes=f.read()
        print(f"[DEBUG] Bytes TIF leídos (Tamaño: {len(tif_bytes)}).");return tif_bytes
    except Exception as e:print(f"[ERROR] TIF temp: {e}");print(traceback.format_exc());st.error(f"Error interno TIF: {e}");return None
    finally:
        if temp_tif_path and os.path.exists(temp_tif_path):
            try:os.remove(temp_tif_path);print(f"[DEBUG] TIF temp eliminado: {temp_tif_path}")
            except Exception as e:print(f"[WARN] No se pudo eliminar TIF temp {temp_tif_path}: {e}")
def create_shapefile_zip_bytes(mask, transform, crs):
    gdf=None;
    try: shapes_gen=rasterio.features.shapes(mask.astype(rasterio.uint8),mask=(mask==1),transform=transform);geometries=[shape(geom) for geom, value in shapes_gen if value==1]
    except Exception as e: print(f"[ERROR] Shapes: {e}"); print(traceback.format_exc()); return None, None
    if not geometries: print("[INFO] No geometries found."); return None, None
    try: gdf=gpd.GeoDataFrame(geometry=geometries,crs=crs); gdf['class_id']=1; print(f"[INFO] GDF created: {len(gdf)} polygons.")
    except Exception as e: print(f"[ERROR] GDF creation: {e}"); print(traceback.format_exc()); return None, None
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_shp=os.path.join(temp_dir,"temp.shp");print(f"[DEBUG] Saving temp SHP: {temp_shp}")
            gdf.to_file(temp_shp,driver='ESRI Shapefile',encoding='utf-8');print(f"[DEBUG] Temp SHP saved.")
            zip_buf=io.BytesIO()
            with zipfile.ZipFile(zip_buf,'w',zipfile.ZIP_DEFLATED) as zf:
                print("[DEBUG] Zipping files...")
                files_to_add={"predicted_mask.shp":temp_shp.replace(".shp",".shp"),"predicted_mask.shx":temp_shp.replace(".shp",".shx"),"predicted_mask.dbf":temp_shp.replace(".shp",".dbf"),"predicted_mask.prj":temp_shp.replace(".shp",".prj")}
                for arcname,filepath in files_to_add.items():
                    if os.path.exists(filepath):print(f"[DEBUG] Adding {arcname}...");zf.write(filepath,arcname=arcname)
                    elif arcname!=".prj":print(f"[WARN] Missing {filepath}")
                    elif gdf.crs:print(f"[WARN] Missing {filepath} (.prj)")
                    else:print(f"[DEBUG] Missing {filepath} (.prj), esperado (sin CRS)")
            zip_buf.seek(0);print("[INFO] ZIP created.");return zip_buf.read(),gdf
    except Exception as e:print(f"[ERROR] Tempfile/Zip: {e}");print(traceback.format_exc());return None,gdf
def calculate_total_tiles(uploaded_files, tile_size=TILE_SIZE, overlap=OVERLAP):
    grand_total = 0; step = max(1, tile_size - overlap); print(f"[INFO] Calculando tiles totales para {len(uploaded_files)} archivo(s)...")
    calc_progress_placeholder = st.empty(); calc_progress = calc_progress_placeholder.progress(0.0, text="Calculando tamaño lote...")
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            uploaded_file.seek(0) # Asegurar que se pueda leer de nuevo
            with rasterio.open(io.BytesIO(uploaded_file.getvalue())) as src: h, w = src.height, src.width; n_tiles_x = math.ceil(w / step); n_tiles_y = math.ceil(h / step); grand_total += (n_tiles_x * n_tiles_y)
            calc_progress.progress(float(i+1)/len(uploaded_files), text=f"Calculando: {uploaded_file.name}")
        except Exception as e: st.warning(f"No se pudo leer {uploaded_file.name}: {e}")
    calc_progress.empty(); print(f"[INFO] Total tiles estimados: {grand_total}"); return max(1, grand_total)

# --- Inicializar Estado de Sesión ---
if 'processing_mode' not in st.session_state: st.session_state.processing_mode = "Procesar Imagen Única"
if 'single_image_results' not in st.session_state: st.session_state.single_image_results = None
if 'batch_processing_results' not in st.session_state: st.session_state.batch_processing_results = []
if 'run_batch_processing' not in st.session_state: st.session_state.run_batch_processing = False
if 'run_single_processing' not in st.session_state: st.session_state.run_single_processing = False
if 'last_uploaded_single_filename' not in st.session_state: st.session_state.last_uploaded_single_filename = None
if 'last_uploaded_batch_filenames' not in st.session_state: st.session_state.last_uploaded_batch_filenames = []
# Guardar selección de mapa base en estado
if 'basemap_selection' not in st.session_state: st.session_state.basemap_selection = BASEMAP_OPTIONS[0] # Default

# --- Interfaz de Streamlit ---
st.title("Segmentación Satelital (Único / Lote)")
st.write(""" Carga imágenes GeoTIFF (4 bandas). Elige modo, modelo, umbral y mapa base. Procesa una imagen o un lote y visualiza/descarga resultados. """)

# --- Barra Lateral (Sidebar) ---
st.sidebar.title("Configuración");
# ---- 0. Selección de Modo ----
st.sidebar.subheader("0. Modo de Procesamiento")
mode_options = ("Procesar Imagen Única", "Procesar Lote de Imágenes")
current_mode_selection = st.sidebar.radio("Selecciona el modo:", options=mode_options, index=mode_options.index(st.session_state.processing_mode), key="processing_mode_selector")
if current_mode_selection != st.session_state.processing_mode:
    st.session_state.processing_mode = current_mode_selection; st.session_state.single_image_results = None; st.session_state.batch_processing_results = []
    st.session_state.run_batch_processing = False; st.session_state.run_single_processing = False
    st.session_state.last_uploaded_single_filename = None; st.session_state.last_uploaded_batch_filenames = []; st.rerun()
st.sidebar.markdown("---"); st.sidebar.info(f"Modo Actual: **{st.session_state.processing_mode}**"); st.sidebar.markdown("---")
# ---- 1. Selección de Modelo ----
st.sidebar.subheader("1. Selección de Modelo"); model_source = st.sidebar.radio("Elige fuente:", ("Modelo por Defecto", "Subir Modelo (.pth)"), key="model_source_radio")
model_to_use = None; uploaded_model_file = None
if model_source == "Modelo por Defecto": model_to_use = load_default_model(DEFAULT_MODEL_PATH)
else:
    uploaded_model_file = st.sidebar.file_uploader("Carga .pth", type=["pth"], key="model_uploader", help="Compatible con ResUNetA.")
    if uploaded_model_file is not None: model_bytes = uploaded_model_file.getvalue(); model_to_use = load_uploaded_model(model_bytes)
if model_to_use:
    if model_source == "Modelo por Defecto": st.sidebar.success("✔️ Usando modelo por defecto.")
    elif uploaded_model_file: st.sidebar.success(f"✔️ Usando: {uploaded_model_file.name}")
    st.sidebar.info("Modelo listo.")
else: st.sidebar.warning("⚠️ Modelo no cargado/inválido.")
# ---- 2. Parámetros de Predicción ----
st.sidebar.markdown("---"); st.sidebar.subheader("2. Parámetros de Predicción")
threshold_slider = st.sidebar.slider("Umbral", 0.1, 0.9, threshold_default, 0.05, key="threshold_slider", help="Probabilidad mínima.")
threshold_current = threshold_slider

st.sidebar.markdown("---"); st.sidebar.info(f"Dispositivo: **{str(device).upper()}**")
st.sidebar.markdown("---"); st.sidebar.subheader("Ayuda Rápida"); st.sidebar.info(""" * Sube GeoTIFFs 4 bandas. * Elige modelo/umbral/mapa. * Selecciona modo. * Pulsa 'Procesar'. * Explora/descarga. """)
# --- FIN Sidebar ---

# --- Sección Principal ---
st.markdown("---")
accept_multiple = (st.session_state.processing_mode == "Procesar Lote de Imágenes"); file_uploader_label = "1. Selecciona imágenes (Lote):" if accept_multiple else "1. Selecciona una imagen:"
uploaded_files = st.file_uploader(file_uploader_label, type=["tif", "tiff"], accept_multiple_files=accept_multiple, key="imageUploader", disabled=(model_to_use is None))

# Lógica para limpiar resultados si cambian archivos
current_uploaded_filenames = [];
if uploaded_files:
    if isinstance(uploaded_files, list): current_uploaded_filenames = sorted([f.name for f in uploaded_files])
    else: current_uploaded_filenames = [uploaded_files.name]
if st.session_state.processing_mode == "Procesar Imagen Única":
    if current_uploaded_filenames and current_uploaded_filenames != st.session_state.last_uploaded_single_filename:
        print("[INFO] Nuevo archivo único detectado."); st.session_state.single_image_results = None; st.session_state.run_single_processing = False; st.session_state.last_uploaded_single_filename = current_uploaded_filenames
elif st.session_state.processing_mode == "Procesar Lote de Imágenes":
     if current_uploaded_filenames and current_uploaded_filenames != st.session_state.last_uploaded_batch_filenames:
        print("[INFO] Nuevos archivos de lote detectados."); st.session_state.batch_processing_results = []; st.session_state.run_batch_processing = False; st.session_state.last_uploaded_batch_filenames = current_uploaded_filenames

# Botones de Inicio
process_button_placeholder = st.empty()
if uploaded_files and model_to_use is not None:
    show_button = True
    if st.session_state.processing_mode == "Procesar Imagen Única" and st.session_state.single_image_results and st.session_state.single_image_results['status'] != 'Error': show_button = False
    elif st.session_state.processing_mode == "Procesar Lote de Imágenes" and st.session_state.batch_processing_results: show_button = False
    if show_button:
        st.markdown("---"); label = "2. Procesar Imagen Seleccionada" if st.session_state.processing_mode == "Procesar Imagen Única" else "2. Iniciar Procesamiento del Lote"
        if process_button_placeholder.button(label, key="start_processing_button"):
            if st.session_state.processing_mode == "Procesar Imagen Única": st.session_state.single_image_results = None; st.session_state.run_single_processing = True; st.session_state.run_batch_processing = False
            else: st.session_state.batch_processing_results = []; st.session_state.run_single_processing = False; st.session_state.run_batch_processing = True
            st.rerun()

# =================================================
# --- Bloque de Procesamiento: IMAGEN ÚNICA ---
# =================================================
if st.session_state.run_single_processing and uploaded_files and model_to_use is not None:
    # (Lógica de procesamiento sin cambios, excepto que no necesita generar GDF dos veces)
    single_uploaded_file = uploaded_files[0] if isinstance(uploaded_files, list) else uploaded_files; file_name_single = single_uploaded_file.name
    st.header(f"Procesando: {file_name_single}"); print(f"\n[INFO] Iniciando proc. único: {file_name_single}")
    progress_bar_single_placeholder = st.empty(); progress_bar_single = None; error_single = None
    try:
        bytes_data_single = single_uploaded_file.getvalue()
        with rasterio.open(io.BytesIO(bytes_data_single)) as src:
            profile=src.profile; transform=src.transform; crs=src.crs; height=src.height; width=src.width; count=src.count
            try: dtype_raster=src.dtypes[0]
            except IndexError: raise ValueError("Error leyendo dtypes.")
            print(f"[INFO] Metadatos: {height}x{width} px, {count} bandas, {dtype_raster}, CRS: {crs}")
            if count!=4: raise ValueError(f"Se esperan 4 bandas, detectadas: {count}.")
            if crs is None: print("[WARN] Imagen sin CRS. Asumiendo EPSG:4326."); crs=rasterio.crs.CRS.from_epsg(4326); profile['crs']=crs; profile['transform']=transform
            else: print(f"[INFO] CRS detectado: {crs.to_string()}")
            progress_bar_single = progress_bar_single_placeholder.progress(0.0, text="Iniciando tiles...")
            step = TILE_SIZE - OVERLAP; step = max(1, step); n_tiles_x = math.ceil(width / step); n_tiles_y = math.ceil(height / step); total_tiles_single = max(1, n_tiles_x * n_tiles_y)
            with st.spinner(f"Procesando {width}x{height} por tiles..."):
                final_mask_single, _ = predict_large_image_tiled(src,model_to_use,device,threshold_current,mean,std,TILE_SIZE,OVERLAP, progress_bar=progress_bar_single, total_tiles_overall=total_tiles_single, processed_offset=0)
            progress_bar_single.progress(1.0, text="¡Tiles completados!")
            st.success(f"Análisis para **{file_name_single}** finalizado.")
            gdf_single = None # Inicializar GDF
            tif_bytes_single = None # Inicializar TIF bytes
            shp_zip_bytes_single = None # Inicializar SHP ZIP bytes
            if final_mask_single is not None:
                 # Generar TIF y SHP bytes aquí para tenerlos listos para session_state y descarga
                 mask_profile_single={'driver':'GTiff','height':final_mask_single.shape[0],'width':final_mask_single.shape[1],'count':1,'dtype':rasterio.uint8,'crs':crs,'transform':transform,'nodata':None}
                 tif_bytes_single = create_mask_tif_bytes(final_mask_single, mask_profile_single)
                 shp_zip_bytes_single, gdf_single = create_shapefile_zip_bytes(final_mask_single, transform, crs)
            # Almacenar resultados
            st.session_state.single_image_results = {
                'input_filename': file_name_single, 'status': 'Completado',
                'mask_tif_bytes': tif_bytes_single, # Guardar bytes TIF
                'shp_zip_bytes': shp_zip_bytes_single, # Guardar bytes SHP ZIP
                'gdf': gdf_single, # Guardar GDF para el mapa
                'profile': profile, 'transform': transform, 'crs': crs, # Guardar metadatos necesarios
                'width': width, 'height': height, 'image_bytes': bytes_data_single, 'error_message': None
            }
    except Exception as e:
        error_msg = f"Error procesando imagen: {e}"; print(f"[ERROR] Procesando {file_name_single}: {traceback.format_exc()}"); st.error(error_msg)
        st.session_state.single_image_results = {'status': 'Error', 'error_message': error_msg, 'input_filename': file_name_single}
        progress_bar_single_placeholder.empty()
    st.session_state.run_single_processing = False; st.rerun()

# ==============================================
# --- Bloque de Procesamiento: LOTE ---
# ==============================================
elif st.session_state.run_batch_processing and uploaded_files and model_to_use is not None:
    # (Lógica interna del bucle de lote sin cambios)
    total_files_in_batch = len(uploaded_files); st.header("Procesando Lote...")
    placeholder_calc = st.empty(); placeholder_calc.info("Calculando tamaño total del lote...")
    grand_total_tiles = calculate_total_tiles(uploaded_files, TILE_SIZE, OVERLAP); placeholder_calc.empty()
    overall_progress_placeholder = st.empty(); overall_progress = overall_progress_placeholder.progress(0.0, text="Iniciando lote...")
    status_placeholder = st.empty(); processed_tiles_offset = 0
    for i, uploaded_file in enumerate(uploaded_files):
        file_name=uploaded_file.name; base_name=os.path.splitext(file_name)[0]; current_file_msg=f"Archivo {i+1}/{total_files_in_batch}: **{file_name}**"
        status_placeholder.info(current_file_msg); print(f"\n[INFO] Iniciando {current_file_msg}")
        gdf=None; final_large_mask=None; error_msg=None; tif_bytes=None; shp_zip_bytes=None; image_bytes_for_thumb=None; width=0; height=0; count=0; profile={}; transform=None; crs=None; tiles_in_this_file=0
        try:
            bytes_data=uploaded_file.getvalue(); image_bytes_for_thumb=bytes_data
            with rasterio.open(io.BytesIO(bytes_data)) as src:
                profile=src.profile; transform=src.transform; crs=src.crs; height=src.height; width=src.width; count=src.count
                try: dtype_raster=src.dtypes[0]
                except IndexError: raise ValueError("Error leyendo dtypes.")
                print(f"[INFO] Metadatos: {height}x{width} px, {count} bandas, {dtype_raster}, CRS: {crs}")
                if count!=4: raise ValueError(f"Se esperan 4 bandas, detectadas: {count}.")
                if crs is None: print("[WARN] Imagen sin CRS. Asumiendo EPSG:4326."); crs=rasterio.crs.CRS.from_epsg(4326); profile['crs']=crs; profile['transform']=transform
                else: print(f"[INFO] CRS detectado: {crs.to_string()}")
                with st.spinner(f"Analizando {file_name} por tiles..."):
                    final_large_mask, tiles_in_this_file = predict_large_image_tiled(src,model_to_use,device,threshold_current,mean,std,TILE_SIZE,OVERLAP, progress_bar=overall_progress, total_tiles_overall=grand_total_tiles, processed_offset=processed_tiles_offset)
                if final_large_mask is not None:
                    mask_profile={'driver':'GTiff','height':final_large_mask.shape[0],'width':final_large_mask.shape[1],'count':1,'dtype':rasterio.uint8,'crs':crs,'transform':transform,'nodata':None}
                    tif_bytes=create_mask_tif_bytes(final_large_mask,mask_profile)
                    if not tif_bytes: error_msg="Fallo al generar TIF"
                    shp_zip_bytes,gdf=create_shapefile_zip_bytes(final_large_mask,transform,crs)
                    if not shp_zip_bytes and (gdf is None or gdf.empty): print(f"[INFO] No polígonos para {file_name}")
                    elif not shp_zip_bytes and gdf is not None: error_msg=error_msg + "; Fallo ZIP SHP" if error_msg else "Fallo ZIP SHP"
                else: error_msg="Fallo en la predicción"
        except Exception as e: print(f"[ERROR] Procesando {file_name}: {traceback.format_exc()}"); error_msg=str(e)
        st.session_state.batch_processing_results.append({'input_filename':file_name,'status':'Error' if error_msg else 'Completado','error_message':error_msg,'image_bytes':image_bytes_for_thumb,'mask_tif_bytes':tif_bytes,'shp_zip_bytes':shp_zip_bytes,'gdf':gdf,'width':width,'height':height,'polygons':len(gdf) if gdf is not None else 0})
        processed_tiles_offset += tiles_in_this_file
    overall_progress.progress(1.0, text=f"¡Lote completado! ({total_files_in_batch} archivos)")
    status_placeholder.success(f"Procesamiento de {total_files_in_batch} archivo(s) completado.")
    st.session_state.run_batch_processing = False


# ==================================================================
# --- Mostrar Resultados (Único o Lote, Leídos de Session State) ---
# ==================================================================

# --- Mostrar resultados de Imagen Única ---
if st.session_state.single_image_results and st.session_state.processing_mode == "Procesar Imagen Única":
    results = st.session_state.single_image_results
    st.markdown("---"); st.header(f"Resultados para: {results.get('input_filename', 'Imagen Procesada')}")
    if results['status'] == 'Completado':
        # Mostrar barra completada (opcional)
        st.progress(1.0, text="Procesamiento completado")
        # ---- Visualización ----
        st.subheader("Resultados Visuales"); col1, col2 = st.columns(2)
        with col1: # Original
            st.write(f"Original ({results.get('width','?') }x{results.get('height','?')})"); img_bytes = results.get('image_bytes')
            if img_bytes:
                try:
                    img_display=None;
                    with rasterio.open(io.BytesIO(img_bytes)) as src_disp:
                        h,w,c=src_disp.height,src_disp.width,src_disp.count; band_indices=[2,1,0]; # AJUSTAR
                        if max(band_indices)>=c: st.error(f"Índices RGB inválidos ({c})")
                        else:
                            if w>MAX_DISPLAY_DIM or h>MAX_DISPLAY_DIM:
                                st.info("Miniatura original.");
                                if OPENCV_AVAILABLE:
                                    aspect=w/h;th=min(MAX_DISPLAY_DIM,h);tw=int(aspect*th);
                                    if tw>MAX_DISPLAY_DIM:tw=MAX_DISPLAY_DIM;th=int(tw/aspect)
                                    img_data=src_disp.read(indexes=[i+1 for i in band_indices],out_shape=(len(band_indices),th,tw),resampling=rasterio.enums.Resampling.bilinear).astype(np.float32)
                                    img_display=denormalize_image(img_data,mean[band_indices],std[band_indices])
                                else: st.warning("OpenCV no disponible.")
                            else: img_data=src_disp.read(indexes=[i+1 for i in band_indices]).astype(np.float32); img_display=denormalize_image(img_data,mean[band_indices],std[band_indices])
                    if img_display is not None: st.image(img_display,caption="Original (RGB Aprox)",use_container_width=True)
                except Exception as e: st.error(f"Error mostrando original: {e}"); print(f"[ERROR] Vis Orig Single: {traceback.format_exc()}")
            else: st.warning("Bytes imagen original no guardados.")
        with col2: # Máscara
            st.write(f"Máscara ({results.get('width','?') }x{results.get('height','?')})"); mask_bytes = results.get('mask_tif_bytes') # Usar TIF bytes para consistencia
            if mask_bytes:
                 try: # Releer máscara desde bytes para display
                     mask_display_array=None
                     with rasterio.open(io.BytesIO(mask_bytes)) as src_mask: mask_array_read=src_mask.read(1)
                     if np.any(mask_array_read): mask_display_array=(mask_array_read*255).astype(np.uint8); print("[DEBUG] Mask array scaled.")
                     else: print("[DEBUG] Mask read from TIF is empty."); mask_display_array=mask_array_read.astype(np.uint8)
                     # Aplicar lógica de miniatura si es necesario
                     if results['width']>MAX_DISPLAY_DIM or results['height']>MAX_DISPLAY_DIM:
                         st.info("Miniatura máscara.");
                         if OPENCV_AVAILABLE:
                             aspect=results['width']/results['height'];th=min(MAX_DISPLAY_DIM,results['height']);tw=int(aspect*th)
                             if tw>MAX_DISPLAY_DIM:tw=MAX_DISPLAY_DIM;th=int(tw/aspect)
                             mask_thumb=cv2.resize(mask_display_array,(tw,th),interpolation=cv2.INTER_NEAREST)
                             st.image(mask_thumb,caption="Miniatura Máscara",use_container_width=True)
                         else: st.warning("OpenCV no disponible.")
                     else: # Mostrar completo
                         st.image(mask_display_array, caption="Máscara Binaria", use_container_width=True)
                 except Exception as e: st.error(f"Error mostrando máscara: {e}"); print(f"[ERROR] Vis Mask Single: {traceback.format_exc()}")
            elif results['status']=='Completado': st.info(f"Máscara vacía (umbral={threshold_current:.2f})."); st.image(np.zeros((min(results['height'],MAX_DISPLAY_DIM//2),min(results['width'],MAX_DISPLAY_DIM//2)),dtype=np.uint8),caption="Vacía",use_container_width=True)
            else: st.warning("Máscara no generada/disponible.")

        # ---- Descargas Individuales ----
        st.subheader("Descargas");
        if results.get('mask_tif_bytes') is not None or results.get('shp_zip_bytes') is not None:
             col_dl1,col_dl2=st.columns(2)
             with col_dl1:
                 st.write("**GeoTIFF:**"); tif_bytes=results.get('mask_tif_bytes');
                 if tif_bytes: tif_fn=f"mask_{os.path.splitext(results['input_filename'])[0]}.tif";st.download_button("⬇️ .tif",tif_bytes,tif_fn,"image/tiff",key="tif_dl_single")
                 else: st.info("No disponible.")
             with col_dl2:
                 st.write("**Shapefile:**"); shp_bytes=results.get('shp_zip_bytes'); gdf_res=results.get('gdf')
                 if shp_bytes and gdf_res is not None and not gdf_res.empty: shp_fn=f"shp_{os.path.splitext(results['input_filename'])[0]}.zip";st.download_button("⬇️ .zip",shp_bytes,shp_fn,"application/zip",key="shp_dl_single")
                 elif gdf_res is None or gdf_res.empty: st.info("No polígonos.")
                 else: st.info("No disponible.") # Si había GDF pero falló el zip
        else: st.warning("Sin máscara, descargas no disponibles.")

        # ---- Mapa Interactivo (Modo Único) ----
        st.subheader("Mapa Interactivo"); gdf_res_map = results.get('gdf')
        if gdf_res_map is not None and not gdf_res_map.empty:
             with st.spinner("🗺️ Generando mapa..."):
                try:
                    gdf_display=None;crs_map=gdf_res_map.crs
                    if crs_map:
                        try:
                            if crs_map!="EPSG:4326":gdf_display=gdf_res_map.to_crs("EPSG:4326")
                            else:gdf_display=gdf_res_map
                        except Exception as e:st.error(f"Error reproyectando:{e}")
                    else:
                        st.warning("GDF sin CRS. Asumiendo EPSG:4326.");
                        try:gdf_res_map.crs="EPSG:4326";gdf_display=gdf_res_map
                        except Exception as e:st.error(f"Error asignando CRS:{e}")
                    if gdf_display is not None:
                        bounds=gdf_display.total_bounds
                        if np.isfinite(bounds).all()and(bounds[2]>bounds[0])and(bounds[3]>bounds[1]):
                            center_y,center_x=(bounds[1]+bounds[3])/2.0,(bounds[0]+bounds[2])/2.0
                            # --- INICIO MODIFICACIÓN MAPA BASE ---
                            selected_tile_name = st.session_state.basemap_selection # Leer del estado
                            tile_info = BASEMAPS.get(selected_tile_name)
                            map_init_tiles = "CartoDB positron" # Fallback
                            if isinstance(tile_info, dict) and isinstance(tile_info.get('tile'), str) : map_init_tiles = tile_info['tile']
                            m=folium.Map(location=[center_y,center_x],zoom_start=12,tiles=map_init_tiles)
                            # Añadir las OTRAS capas base
                            for name, tile_data in BASEMAPS.items():
                                 try:
                                     if isinstance(tile_data.get('tile'), str) and name != selected_tile_name : # Es un nombre de tile incorporado (y no el default)
                                         folium.TileLayer(tile_data['tile'], name=name, attr=tile_data['attr'], control=True).add_to(m)
                                     elif isinstance(tile_data.get('tile'), folium.TileLayer): # Es un objeto TileLayer (Esri)
                                          # Clonar o añadir directamente, asegurar control=True
                                          layer_to_add = tile_data['tile']
                                          # Forzar el nombre del diccionario para el control de capas
                                          layer_to_add.options['name'] = name
                                          layer_to_add.options['control'] = True
                                          layer_to_add.options['overlay'] = False # Asegurar que es capa base
                                          layer_to_add.add_to(m)
                                 except Exception as e_tile: print(f"[ERROR] Añadiendo tile layer '{name}': {e_tile}")
                            # --- FIN MODIFICACIÓN MAPA BASE ---
                            folium.GeoJson(gdf_display,name="Predicción",style_function=lambda x:{"fillColor":"#2ca02c","color":"#006400","weight":1,"fillOpacity":0.6},tooltip=folium.features.GeoJsonTooltip(fields=['class_id'],aliases=['ID:'])).add_to(m)
                            folium.LayerControl().add_to(m) # Añadir control de capas
                            st_folium(m,width='100%',height=600,returned_objects=[],key="map_single") # Key única
                        else:st.warning("Límites inválidos GDF.")
                    else:st.warning("No se pudo preparar GDF para mapa.")
                except Exception as e:st.error(f"Error mapa:{e}");st.code(traceback.format_exc())
        else: st.info("Mapa no disponible (sin polígonos).")
    elif results['status'] == 'Error': st.error(f"Error en procesamiento: {results.get('error_message', 'Desconocido')}")

# --- Mostrar Resumen y Opciones post-procesamiento LOTE ---
elif st.session_state.batch_processing_results and st.session_state.processing_mode == "Procesar Lote de Imágenes":
    st.markdown("---"); st.subheader("Resumen del Procesamiento por Lotes")
    # Mostrar barra de progreso completada (opcional)
    # st.progress(1.0, text="¡Lote completado!")
    summary_data=[{'Archivo':r['input_filename'],'Estado':r['status'],'# Polígonos':r['polygons'],'Error':r['error_message'] if r['error_message'] else '-'} for r in st.session_state.batch_processing_results]
    st.dataframe(pd.DataFrame(summary_data))
    successful_results=[res for res in st.session_state.batch_processing_results if res['status']=='Completado']
    if successful_results:
        st.markdown("---"); st.subheader("Visualización Detallada")
        successful_filenames=[res['input_filename'] for res in successful_results]
        selected_index=st.selectbox("Ver detalles para:",options=range(len(successful_filenames)),format_func=lambda x:successful_filenames[x],key="detail_selector")
        if selected_index is not None:
            selected_result=successful_results[selected_index]
            st.markdown(f"#### Detalles para: `{selected_result['input_filename']}`")
            col1,col2=st.columns(2)
            # (Lógica de visualización selectiva Original - sin cambios)
            with col1:
                st.write(f"Original ({selected_result['width']}x{selected_result['height']})")
                if selected_result['image_bytes']:
                    try:
                        image_display=None
                        with rasterio.open(io.BytesIO(selected_result['image_bytes'])) as src_orig:
                            h_orig,w_orig=selected_result['height'],selected_result['width'];count_orig=src_orig.count
                            band_indices_rgb=[2,1,0]; # AJUSTA ORDEN RGB
                            if max(band_indices_rgb)>=count_orig: st.error(f"Índices RGB inválidos ({count_orig} bandas).")
                            else:
                                if w_orig>MAX_DISPLAY_DIM_PREVIEW or h_orig>MAX_DISPLAY_DIM_PREVIEW:
                                    st.caption("Mostrando miniatura.");aspect_ratio=w_orig/h_orig;thumb_h=min(MAX_DISPLAY_DIM_PREVIEW,h_orig);thumb_w=int(aspect_ratio*thumb_h)
                                    if thumb_w>MAX_DISPLAY_DIM_PREVIEW:thumb_w=MAX_DISPLAY_DIM_PREVIEW;thumb_h=int(thumb_w/aspect_ratio)
                                    img_data=src_orig.read(indexes=[i+1 for i in band_indices_rgb],out_shape=(len(band_indices_rgb),thumb_h,thumb_w),resampling=rasterio.enums.Resampling.bilinear).astype(np.float32)
                                else: img_data=src_orig.read(indexes=[i+1 for i in band_indices_rgb]).astype(np.float32)
                                image_display=denormalize_image(img_data,mean[band_indices_rgb],std[band_indices_rgb])
                        if image_display is not None: st.image(image_display,caption="Vista Previa Original",use_container_width=True)
                    except Exception as e: st.error(f"Error mostrando vista previa original: {e}"); print(f"[ERROR] Display Orig Sel: {traceback.format_exc()}")
                else: st.warning("Bytes de imagen original no disponibles.")
            # (Lógica de visualización selectiva Máscara - sin cambios)
            with col2:
                st.write(f"Máscara ({selected_result['width']}x{selected_result['height']})")
                if selected_result['mask_tif_bytes']:
                    try:
                        print("[DEBUG] Reading mask TIF bytes for display...")
                        mask_display_array=None
                        with rasterio.open(io.BytesIO(selected_result['mask_tif_bytes'])) as src_mask: mask_array_read=src_mask.read(1)
                        if np.any(mask_array_read): mask_display_array=(mask_array_read*255).astype(np.uint8); print("[DEBUG] Mask array scaled.")
                        else: print("[DEBUG] Mask read from TIF is empty."); mask_display_array=mask_array_read.astype(np.uint8)
                        st.image(mask_display_array,caption="Vista Previa Máscara",use_container_width=True)
                    except Exception as e: st.error(f"Error mostrando vista previa máscara: {e}"); print(f"[ERROR] Display Mask Sel: {traceback.format_exc()}")
                elif selected_result['status']=='Completado': st.info("Máscara vacía."); st.image(np.zeros((min(selected_result['height'],MAX_DISPLAY_DIM_PREVIEW//2),min(selected_result['width'],MAX_DISPLAY_DIM_PREVIEW//2)),dtype=np.uint8),caption="Vacía",use_container_width=True)
                else: st.warning("Bytes de máscara no disponibles.")

            # ---- Mapa Interactivo Selectivo (CON MAPAS BASE) ----
            st.write("Mapa Interactivo:")
            selected_gdf = selected_result.get('gdf')
            if selected_gdf is not None and not selected_gdf.empty:
                with st.spinner("🗺️ Generando mapa para archivo seleccionado..."):
                    try:
                        gdf_display=None;crs_map=selected_gdf.crs
                        # (Lógica de reproyección sin cambios)
                        if crs_map:
                            try:
                                if crs_map!="EPSG:4326":gdf_display=selected_gdf.to_crs("EPSG:4326")
                                else:gdf_display=selected_gdf
                            except Exception as e:st.error(f"Error reproyectando:{e}")
                        else:
                            st.warning("GDF sin CRS. Asumiendo EPSG:4326.");
                            try:selected_gdf.crs="EPSG:4326";gdf_display=selected_gdf
                            except Exception as e:st.error(f"Error asignando CRS:{e}")

                        if gdf_display is not None:
                            bounds=gdf_display.total_bounds
                            if np.isfinite(bounds).all()and(bounds[2]>bounds[0])and(bounds[3]>bounds[1]):
                                center_y,center_x=(bounds[1]+bounds[3])/2.0,(bounds[0]+bounds[2])/2.0

                                # --- INICIO MODIFICACIÓN MAPA BASE (LOTE) ---
                                selected_tile_name_lote = st.session_state.basemap_selection # Leer selección
                                tile_info_lote = BASEMAPS.get(selected_tile_name_lote)
                                map_init_tiles_lote = "CartoDB positron" # Fallback
                                if isinstance(tile_info_lote, dict) and isinstance(tile_info_lote.get('tile'), str): map_init_tiles_lote = tile_info_lote['tile']
                                m_lote=folium.Map(location=[center_y,center_x],zoom_start=13,tiles=map_init_tiles_lote) # Usar tile inicial seleccionado

                                # Añadir OTRAS capas base con atribución correcta
                                for name, tile_data in BASEMAPS.items():
                                    # Añadir solo si NO es la que ya está como base inicial O si es Esri (que es un objeto)
                                    is_default = isinstance(tile_data.get('tile'), str) and tile_data.get('tile') == map_init_tiles_lote
                                    if not is_default or isinstance(tile_data.get('tile'), folium.TileLayer):
                                        try:
                                            if isinstance(tile_data.get('tile'), str): # OSM, Stamen, CartoDB
                                                folium.TileLayer(tile_data['tile'], name=name, attr=tile_data['attr'], control=True).add_to(m_lote)
                                            elif isinstance(tile_data.get('tile'), folium.TileLayer): # Esri
                                                layer_to_add = tile_data['tile']
                                                layer_to_add.options['name'] = name # Asegurar nombre correcto en control
                                                layer_to_add.options['control'] = True
                                                layer_to_add.add_to(m_lote)
                                        except Exception as e_tile: print(f"[ERROR] Añadiendo tile layer (lote) '{name}': {e_tile}")
                                # --- FIN MODIFICACIÓN MAPA BASE (LOTE) ---

                                folium.GeoJson(gdf_display,name="Predicción",style_function=lambda x:{"fillColor":"#2ca02c","color":"#006400","weight":1,"fillOpacity":0.6},tooltip=folium.features.GeoJsonTooltip(fields=['class_id'],aliases=['ID:'])).add_to(m_lote)
                                folium.LayerControl().add_to(m_lote) # Añadir control de capas
                                st_folium(m_lote,width='100%',height=450,returned_objects=[],key=f"map_batch_{selected_index}")
                            else:st.warning("Límites inválidos GDF.")
                        else:st.warning("No se pudo preparar GDF para mapa.")
                    except Exception as e:st.error(f"Error mapa:{e}");st.code(traceback.format_exc())
            else: st.info("Mapa no disponible (sin polígonos).")
    # Descarga Agrupada
    # (Sin cambios)
    if successful_results:
        st.markdown("---"); st.subheader("Descarga Agrupada")
        files_to_zip=[]
        for res in successful_results:
            base_name=os.path.splitext(res['input_filename'])[0]
            if res['mask_tif_bytes']:files_to_zip.append({'name':f"{base_name}_mask.tif",'bytes':res['mask_tif_bytes']})
            if res['shp_zip_bytes']:files_to_zip.append({'name':f"{base_name}_shapefile.zip",'bytes':res['shp_zip_bytes']})
        if files_to_zip:
            final_zip_buffer=io.BytesIO();
            with zipfile.ZipFile(final_zip_buffer,'w',zipfile.ZIP_DEFLATED) as final_zip:
                for item in files_to_zip:final_zip.writestr(item['name'],item['bytes'])
            final_zip_buffer.seek(0); st.download_button(label=f"⬇️ Descargar {len(files_to_zip)} Archivos (.zip)",data=final_zip_buffer,file_name="resultados_lote.zip",mime="application/zip")
        else:st.warning("No hay archivos válidos para descargar.")
    elif 'run_batch_processing' in st.session_state and not st.session_state.run_batch_processing and not successful_results and len(st.session_state.batch_processing_results) > 0: st.warning("No hubo resultados exitosos.")


# ---- MENSAJES DE ESTADO INICIAL O DE ESPERA ----
# (Sin cambios)
elif model_to_use is None and uploaded_files: st.warning("⬅️ Archivo(s) cargado(s), pero falta modelo.")
elif not uploaded_files and model_to_use is not None: st.info(f"Modelo listo. Carga {'imágenes' if st.session_state.processing_mode == 'Procesar Lote de Imágenes' else 'una imagen'} GeoTIFF.")
elif model_to_use is None and not uploaded_files: st.info("⬅️ Bienvenido. Selecciona/carga modelo y luego imagen(es).")