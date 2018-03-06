import sys
def read_image_names(file_path):
    with open(file_path) as f:
        image_paths = f.read().splitlines()

        return image_paths

def gen_parilist(file_path, saved_name):
    image_paths = read_image_names(file_path)
    with open(saved_name, 'w') as f:
        for i in range(0, len(image_paths), 2):
            pair = image_paths[i] + ' ' + image_paths[i + 1] + '\n'
            f.write(pair)

    return True

if __name__ == '__main__':
   file_path  = sys.argv[1]
   saved_name = sys.argv[2]

   gen_parilist(file_path, saved_name)
