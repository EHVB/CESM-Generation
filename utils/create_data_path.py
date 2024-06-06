import os
def create_data_path(input_dir,output_dir, extensions=['.png', '.jpg', '.jpeg']):
    
    def is_image_file(filename):
        return any(filename.lower().endswith(ext) for ext in extensions)

    output_files = [f for f in os.listdir(output_dir) if is_image_file(f)]
    output_paths = [os.path.join(output_dir, f) for f in output_files]
    input_paths = [os.path.join(input_dir, modify_filename(f)) for f in output_files]

    return input_paths, output_paths

def modify_filename(filename):
    # Split the filename on underscores
    parts = filename.split('_')
    
    # Check if the last part contains 'C'
    if 'C' in parts[-1]:
        # Modify the last part by replacing 'C' with 'MG'
        parts[-1] = parts[-1].replace('C', 'MG')
    else:
        parts[-2] = parts[-2].replace('C', 'MG')
    
    # Join the parts back together with underscores
    new_filename = '_'.join(parts)
    
    return new_filename
    