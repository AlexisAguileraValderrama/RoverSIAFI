import sys
import bpy

import random

argv = sys.argv
argv = argv[argv.index("--") + 1:]

# input_file = argv[0]
file_name = argv[0]
path_files = argv[1]
axis_up = 'Z'

seed = random.randint(0,10000)
print(seed)

bpy.ops.mesh.landscape_add(random_seed = seed, 
                           mesh_size_x=10,
                           mesh_size_y=10,
                           noise_type = 'hetero_terrain',
                           noise_offset_x=0,
                           noise_size_x=1,
                           dimension=1.36, 
                           height=0.68, 
                           height_offset=0, 
                           falloff_x=50, 
                           falloff_y=50, 
                           maximum=0.31, 
                           minimum=-0.22, refresh=True)

# print('Input file:', input_file)
print('Path of files:',path_files)
print('obj Output file:', file_name+".obj")
print('dae Output file:', file_name+".dae")
print('Converting with blender...')

# Importing and exporting the scene
bpy.ops.wm.obj_export(filepath=path_files+"/obj/"+file_name+".obj", up_axis=axis_up)
bpy.ops.wm.collada_export(filepath=path_files+"/dae/"+file_name+".dae",export_global_up_selection=axis_up)

print('-' * 30)
print('  Converting done !')
print('-' * 30)
