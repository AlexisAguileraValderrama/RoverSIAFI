#!/bin/bash

# Convert a Gazebo world to an octomap file
# The .bt file can later be imported in ROS for the Octomap node

worlds_num=$1
vox_size=$2

__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -e $HOME/binvox ]; then
  echo "Unable to locate the binvox executable in the $HOME folder !"
fi

echo "directorio: $__dir"

# Define an array of subfolder names
SUBFOLDERS=("$__dir/obj" "$__dir/dae" "$__dir/binvox" "$__dir/bt")

# Loop through each subfolder
for SUBFOLDER in "${SUBFOLDERS[@]}"; do
  # Check if the subfolder exists
  if [ ! -d "$SUBFOLDER" ]; then
    # Create the subfolder if it does not exist
    mkdir "$SUBFOLDER"
  fi
done

# Range of numbers
for i in $(seq 0 $worlds_num)
do

  world_name="world_$i"
  echo "Creating world $world_name"

  # Define an array of file names
  FILES=("$__dir/obj/$world_name.obj" "$__dir/obj/$world_name.mtl" "$__dir/dae/$world_name.dae" "$__dir/binvox/$world_name.binvox" "$__dir/bt/$world_name.bt")

  # Loop through each file
  for FILE in "${FILES[@]}"; do
    # Check if the file exists
    if [ -f "$FILE" ]; then
      # Remove the file
      rm "$FILE"
    fi
  done

  # Convert from Collada .dae to .obj using Blender
  blender ${__dir}/terreno.blend --background --python ${__dir}/blend_convert_to_obj.py -- ${world_name} ${__dir}

  # Convert from .obj to voxels with binvox
  # Assuming the binvox executable is in $HOME
  stdbuf -oL -eL $HOME/binvox -e -fit ${__dir}/obj/${world_name}.obj -d ${vox_size} -rotz -rotz -rotz -ri > /dev/null 2>&1
  mv ${__dir}/obj/${world_name}.binvox ${__dir}/binvox/


  # Convert from .binvox to octomap .binvox.bt
  binvox2bt --mark-free ${__dir}/binvox/${world_name}.binvox

  # Move the file to the expected output
  mv ${__dir}/binvox/${world_name}.binvox.bt ${__dir}/bt/${world_name}.bt

done

#octovis ${__dir}/bt/${out_name}.bt