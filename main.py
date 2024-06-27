import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from gooey import Gooey, GooeyParser

@Gooey(program_name="Image Augmentation Tool by Arhan mansoori")
def main():
    parser = GooeyParser(description="Augment Images")
    parser.add_argument('img_path', help='Path to the image to augment', widget='FileChooser')
    parser.add_argument('num_images_to_augment', type=int, help='Total number of images to augment')
    parser.add_argument('rotation_range', type=float, help='Rotation range (degrees)', default=20)
    parser.add_argument('width_shift_range', type=float, help='Width shift range (fraction of total width)', default=0.2)
    parser.add_argument('height_shift_range', type=float, help='Height shift range (fraction of total height)', default=0.2)
    parser.add_argument('shear_range', type=float, help='Shear range (degrees)', default=0.2)
    parser.add_argument('zoom_range', type=float, help='Zoom range (fraction)', default=0.2)
    parser.add_argument('horizontal_flip', choices=['True', 'False'], help='Enable horizontal flip', default='True')
    parser.add_argument('fill_mode', choices=['nearest', 'constant', 'reflect', 'wrap'], help='Fill mode for points outside the boundaries', default='nearest')
    
    args = parser.parse_args()
    
    img_path = args.img_path
    num_images_to_augment = args.num_images_to_augment
    rotation_range = args.rotation_range
    width_shift_range = args.width_shift_range
    height_shift_range = args.height_shift_range
    shear_range = args.shear_range
    zoom_range = args.zoom_range
    horizontal_flip = args.horizontal_flip == 'True'
    fill_mode = args.fill_mode

    # Define image dimensions and format
    target_size = (300, 350)
    img_format = 'jpg'

    # Check if the provided path is valid
    if not os.path.isfile(img_path):
        print("The provided path is not valid.")
        return

    # Create the "augmented_images" folder if it doesn't exist
    augmented_dir = 'augmented_images'
    os.makedirs(augmented_dir, exist_ok=True)

    # Initialize an ImageDataGenerator for augmentation
    datagen = ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        fill_mode=fill_mode
    )

    # Load the image
    img = load_img(img_path)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    # Generate augmented images
    i = 0
    img_filename = os.path.splitext(os.path.basename(img_path))[0]
    for batch in datagen.flow(x, batch_size=1, save_to_dir=augmented_dir, save_prefix=f'{img_filename}_aug', save_format=img_format):
        i += 1
        if i >= num_images_to_augment:
            break

    print("Augmentation completed. Augmented images are saved in the 'augmented_images' folder.")

if __name__ == '__main__':
    main()
