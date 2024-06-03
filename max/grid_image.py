from PIL import Image
import os
import natsort

def combine_images(image_paths_2d: dict, save_path, spacing):
    """
    Combine images into a grid with spacing and save the result.

    Parameters:
    image_paths_2d (list of list of str): 2D list of paths to the images.
    save_path (str): Path to save the combined image.
    spacing (int): Number of pixels for the spacing between images.
    """
    # Flatten the 2D list and open images
    images_2d = [[Image.open(f'plots/{path}') for path in row] for row in image_paths_2d.values()]

    # Resize all images to firist image size
    img_width, img_height = images_2d[0][0].size
    images_2d = [[img.resize((img_width, img_height)) for img in row] for row in images_2d]

    # Calculate grid size
    grid_rows = len(images_2d)
    grid_cols = len(images_2d[0])

    # Calculate the size of the final image including spacing
    total_width = img_width * grid_cols + spacing * (grid_cols - 1)
    total_height = img_height * grid_rows + spacing * (grid_rows - 1)

    # Create a new blank image with the appropriate size
    grid_img = Image.new('RGB', (total_width, total_height), (255, 255, 255))

    # Paste each image into the grid image with spacing
    for row_idx, row in enumerate(images_2d):
        for col_idx, img in enumerate(row):
            x_position = col_idx * (img_width + spacing)
            y_position = row_idx * (img_height + spacing)
            grid_img.paste(img, (x_position, y_position))

    # Save the result
    grid_img.save(save_path)


def filter_and_extract_names(names, search_string):
    """
    Filter out names containing a specific string and extract those names.

    Parameters:
    names (list of str): List of names.
    search_string (str): String to search for in the names.

    Returns:
    tuple: A tuple containing two lists:
           - The filtered list with names containing the search string removed.
           - The list of names that contain the search string.
    """
    # Extract names that contain the search string
    extracted_names = [name for name in names if search_string in name]

    # Filter out names that contain the search string
    filtered_names = [name for name in names if search_string not in name]

    return filtered_names, extracted_names


def get_paths(ds_name):
    all_paths = os.listdir(f'plots/')
    all_paths = [f for f in all_paths if ds_name in f]

    save_imgs = {}
    for model_name in ['True', '125m', 'large', 'DRN', 'GAT', 'MGN']:
        all_paths, model_paths = filter_and_extract_names(all_paths, model_name)
        save_imgs[model_name] = natsort.natsorted(model_paths)
    save_imgs['True'] = natsort.natsorted(all_paths)

    print(save_imgs)
    # ds_paths = [f for f in all_paths if ds_name in f]
    # print(ds_paths)
    return save_imgs


if __name__ == "__main__":
    ds_name = 'cylinder'
    paths = get_paths(ds_name)
    combine_images(paths, f'{ds_name}.png', spacing=20)

# # Example usage:
# image_paths_2d = [
#     ['airfoil_0.png', 'airfoil_20.png'],
#     ['airfoil_40.png', 'airfoil_40.png']
# ]  # Add your image paths here
# save_path = 'combined_image.png'
# combine_images(image_paths_2d, save_path, spacing=10)
