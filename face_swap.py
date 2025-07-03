import os
import cv2
import roop.globals
from roop.FaceSet import FaceSet
from roop.ProcessEntry import ProcessEntry
from roop.core import batch_process_regular
from roop.face_util import get_all_faces
from settings import Settings

# --- CONFIG ---
SOURCE_IMAGE = 'source.jpg'  # The face to swap in
TARGET_IMAGE = 'target.jpg'  # The image to swap face in
OUTPUT_IMAGE = 'output.jpg'  # The result

# --- 1. Load images ---
source_img = cv2.imread(SOURCE_IMAGE)
target_img = cv2.imread(TARGET_IMAGE)

if source_img is None or target_img is None:
    raise FileNotFoundError('Could not load source or target image.')

# --- 2. Extract faces ---
source_faces = get_all_faces(source_img)
target_faces = get_all_faces(target_img)

if not source_faces or not target_faces:
    raise ValueError('No face found in source or target image.')

# --- 3. Create FaceSets ---
source_faceset = FaceSet()
source_faceset.faces = [source_faces[0]]  # Only the first face
roop.globals.INPUT_FACESETS = [source_faceset]

target_faceset = FaceSet()
target_faceset.faces = [target_faces[0]]
roop.globals.TARGET_FACES = [target_faceset]

# --- 4. Create ProcessEntry for the target image ---
entry = ProcessEntry(TARGET_IMAGE, 0, 0, 0)
files = [entry]

# --- 5. Set required globals ---
roop.globals.distance_threshold = 1.0
roop.globals.blend_ratio = 0.5
roop.globals.face_swap_mode = 'first'
if roop.globals.CFG is None:
    roop.globals.CFG = Settings('config.yaml')
roop.globals.output_path = '.'

# --- 6. Run face swap ---
batch_process_regular(
    swap_model='insightface',
    output_method='Files',
    files=files,
    masking_engine='',
    new_clip_text='',
    use_new_method=True,
    imagemask=None,
    restore_original_mouth=False,
    num_swap_steps=1,
    progress=None,
    selected_index=0
)

# --- 7. Move result to output.jpg ---
# The output will be in the same folder as the target, with a modified name
import glob
result_files = glob.glob(f"{os.path.splitext(TARGET_IMAGE)[0]}*jpg")
if result_files:
    os.rename(result_files[0], OUTPUT_IMAGE)
    print(f"Face swapped! Output saved as {OUTPUT_IMAGE}")
else:
    print("Face swap failed: No output file found.") 