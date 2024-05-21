import os
def create_data_path(input_dir,output_dir,  extensions=['.png', '.jpg', '.jpeg']):

    def is_image_file(filename):
        return any(filename.lower().endswith(ext) for ext in extensions)

    input_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if is_image_file(f)]
    output_paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if is_image_file(f)]

    return input_paths, output_paths
