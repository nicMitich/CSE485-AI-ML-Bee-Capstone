import bpy, random, math, os
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view


print([obj.name for obj in bpy.data.objects if obj.type == 'ARMATURE'])
bee = bpy.data.objects["RootNode.0"]
print("Rotation mode:", bee.rotation_mode)
print("Children:", [child.name for child in bee.children])



# ========== CONFIG ==========
output_dir = r"C:\ASU\Capstone\synthdataset"  # Change this to desired folder
num_frames = 100                 # Number of rendered frames
shift_range = 5.0                # Bee shift range in X/Y (CHANGE THIS FOR MORE LEFT/RIGHT MOVEMENT)
rotation_range = 40              # Rotation range in degrees (not sure if this works)
bee_name = "RootNode.0"     # EXACT name of your bee object in blender outliner, might vary so check before compilation

# =============================
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE_NEXT"
scene.render.image_settings.file_format = "PNG"
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100

bee = bpy.data.objects.get(bee_name)
if not bee:
    raise Exception(f"Object '{bee_name}' not found!")
    
bee.rotation_mode = 'XYZ' 
    
# --- Show bounding box in viewport ---
bee.display_type = 'BOUNDS'     # Shows only the bounding box
bee.show_bounds = True          # Ensures the bounds are visible

cam = scene.camera
if not cam:
    raise Exception("No active camera found! Set a camera in your scene.")

# --- Helper: project 3D object bbox to 2D camera coordinates ---
def bbox_2d_yolo(obj, cam, scene):
    mat = obj.matrix_world
    corners = [Vector(corner) for corner in obj.bound_box]
    coords = [world_to_camera_view(scene, cam, mat @ c) for c in corners]
    # Filter only visible coords
    coords = [c for c in coords if 0.0 <= c.x <= 1.0 and 0.0 <= c.y <= 1.0 and c.z >= 0.0]
    if not coords:
        return None
    xs = [c.x for c in coords]
    ys = [c.y for c in coords]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    # YOLO normalized: x_center y_center width height
    x_c = (minx + maxx) / 2
    y_c = (miny + maxy) / 2
    w = (maxx - minx)
    h = (maxy - miny)
    return (x_c, y_c, w, h)

# --- Store original transform ---
orig_loc = bee.location.copy()
orig_rot = bee.rotation_euler.copy()

# --- Main render loop ---
for i in range(num_frames):
    random.seed(i*999)

    # Randomize bee transform
    bee.location.x = orig_loc.x + random.uniform(-shift_range, shift_range)
    bee.location.y = orig_loc.y + random.uniform(-shift_range, shift_range)
    bee.rotation_euler.z = orig_rot.z + math.radians(random.uniform(-rotation_range, rotation_range))
    
    # Render frame
    img_path = os.path.join(output_dir, "images", f"frame_{i:05d}.png")
    scene.render.filepath = img_path
    bpy.ops.render.render(write_still=True)

    # Generate YOLO label
    label_path = os.path.join(output_dir, "labels", f"frame_{i:05d}.txt")
    bbox = bbox_2d_yolo(bee, cam, scene)
    if bbox:
        x_c, y_c, w, h = bbox
        # For single class (worker bee) use class_id = 0
        with open(label_path, "w") as f:
            f.write(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
            
# --- Reset bee to original position and rotation ---
bee.location = orig_loc
bee.rotation_euler = orig_rot
bpy.context.view_layer.update()

print(f"Rendered {num_frames} frames with YOLO labels to {output_dir}")
