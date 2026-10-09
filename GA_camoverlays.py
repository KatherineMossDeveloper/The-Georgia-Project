# GA_camoverlays.py                           Oct. 1 version, using bicubic
# colors are here:  https://matplotlib.org/stable/users/explain/colors/colormaps.html
#
# Calling structure.
#
# camoverlays_driver
#  generate_images_orig224()  create a set of the original images, but with 224x224 dimensions.
#  get_model()                create a ResNet101 model, using the weights file created in GA.
#  get_cam_color_scheme()     create a color scheme
#  run_layercam()             drive the creation of CAM images and CAM overlay images.
#    load_image()             load images
#    predict_class_index()    get the class designation:  CEX or PG
#    layercam_multi()         loop through layers
#      layercam_single_layer() get each layer CAM
#    cam_percentile_bands()   apply a different color to each band per % weight
#    to_heatmap()
#    combine_heatmaps()
#    overlay_heatmap()
#
# Notes.
import traceback
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
from GAutility import get_model, get_cam_color_scheme
from tensorflow.keras.applications.resnet50 import preprocess_input

# color scheme
blue = get_cam_color_scheme("RdBu")
purple = get_cam_color_scheme("Purples")
pink = get_cam_color_scheme("gist_rainbow")


def generate_images_orig224(source_folder, output_folder):

    source_folder = Path(source_folder)
    output_folder = Path(output_folder)

    # Create the destination folder if it doesn't exist.
    output_folder.mkdir(parents=True, exist_ok=True)

    # Process the original images.
    for image_path in source_folder.iterdir():
        if image_path.suffix.lower() != ".png":
            continue

        # Load, convert to RGB, and resize using bicubic interpolation.
        with Image.open(image_path) as original:
            img = original.convert("RGB")    # convert to RBG & remove the A channel.
            img = img.resize((224, 224), Image.Resampling.BICUBIC)
            # Save the resized image using its original filename.
            img.save(output_folder / image_path.name)

            print(f"Resized: {image_path.name}")

    print("Finished generating images_orig224.")


def load_image(path):

    # Open the image.
    image = Image.open(path).convert("RGB")
    # Prepare a separate copy for ResNet.
    image_array_float = np.asarray(image, dtype=np.float32)
    # add a new dimension, a batch number @ front:  (224,224,3) -> (1,224,224,3).
    image_array_float_batch_no = np.expand_dims(image_array_float, axis=0)
    # do Keras preprocess, the same as when training.
    image_array_preprocessed = preprocess_input(image_array_float_batch_no)

    return image, image_array_preprocessed


# example:  heatmap_70_80_array = to_heatmap(cam=band_70_80, cmap=blue, alpha_scale=0.7)
def to_heatmap(cam, cmap, alpha_scale=1.0):

    cam = np.nan_to_num(cam)           # replace NaNs with numbers.
    cam = cam / (cam.max() + 1e-8)     # scale image array 0 to 1.  1e-8 prevents 0 div. err.
    cam = np.clip(cam, 0, 1)

    rgba = cmap(cam)                   # add color by adding RGBA dimensions
    rgba[..., 3] = cam * alpha_scale   # apply the transparency to alpha channel (3)

    return (rgba * 255).astype(np.uint8)  # convert to RGBA values, 0 - 255


# overlay the CAM heatmap onto the original image.
def overlay_heatmap(base_img_pil, heatmap_rgba):

    # Convert the heatmap array to a PIL image and resize it
    # so that it is same size as the original image.
    heat = Image.fromarray(heatmap_rgba, mode="RGBA").resize(
        base_img_pil.size, Image.BICUBIC)

    # Add an alpha channel to the original image.
    base = base_img_pil.convert("RGBA")

    # Combine the original image and transparent CAM heatmap.
    return Image.alpha_composite(base, heat)


# Generate a LayerCAM for one selected convolutional layer.
def layercam_single_layer(model, image_array, class_index, layer_name):

    # 0. Prepare to get the selected convolutional layer and
    #    the final classification layer.
    target_layer = model.get_layer(layer_name).output
    output_layer = model.layers[-1]

    # 1. Build a temporary submodel that returns both...
    #    - the selected layer's feature maps
    #    - the final Dense classifier vector; info used to create class score.
    submodel = tf.keras.Model(
        inputs=model.inputs,
        outputs=[target_layer, output_layer.input])

    # 2. Run a forward pass while recording the calculations
    #    needed to compute gradients.
    with tf.GradientTape() as tape:

        activations, final_features = submodel(image_array, training=False)

        # Reconstruct the final Dense layer's logit using its
        # existing trained weights and bias.
        logits = tf.linalg.matmul(final_features, output_layer.kernel)

        if output_layer.use_bias:
            logits = tf.nn.bias_add(logits, output_layer.bias)

        # Positive logit favors PG; negative logit favors CEX.
        # Reverse the score for CEX so that positive gradients
        # represent evidence supporting the requested class.
        score = (logits[:, 0] if class_index == 1 else -logits[:, 0])

    # 3. Calculate how the class score changes with respect
    #    to each activation in the selected convolutional layer.
    grads = tape.gradient(score, activations)
    print("Logit:", logits.numpy())

    # 4. LayerCAM uses positive gradients as local importance weights.
    grads = tf.nn.relu(grads)

    # 5. Weight each feature activation by its local gradient,
    #    then sum across all feature-map channels.
    cam = tf.reduce_sum(grads * activations, axis=-1)

    # 6. Keep positive class-supporting attribution.
    cam = tf.nn.relu(cam)

    # Remove the batch dimension and convert to a NumPy array.
    cam = cam[0].numpy()

    # Normalize the LayerCAM to 0–1.
    if cam.max() > 0:
        cam = cam / cam.max()

    return cam


# Generate LayerCAMs for the selected CNN layers and combine them.
def layercam_multi(model, image_array, class_index, layer_names, img_size):

    cams = []

    # Generate a CAM for each selected CNN layer.
    for layer_name in layer_names:
        cam = layercam_single_layer(
            model, image_array, class_index, layer_name)

        # Resize each CAM to a common size, so they can be combined.
        # "resize" needs a 3rd dim, so add it temporarily.
        # cam[..., None]    adds a channel dimension
        # .numpy()[..., 0]  takes out the channel dimension.
        cam = tf.image.resize(
            cam[..., None],
            img_size,
            method="bilinear"
        ).numpy()[..., 0]

        cams.append(cam)

    # average the CAMs when more than one layer is selected.
    cam = np.mean(cams, axis=0)

    # normalize the combined CAM to the range 0–1.
    cam = cam / (cam.max() + 1e-8)

    return cam


def predict_class_index(model, image_array):

    # Ensure 4D float32 [B,H,W,C] in [0,1]
    if isinstance(image_array, np.ndarray):
        image_array = tf.convert_to_tensor(image_array, dtype=tf.float32)
    else:
        image_array = tf.cast(image_array, tf.float32)

    # Forward pass (no .numpy() here)
    y = model(image_array, training=False)

    # If the model returns multiple outputs, take the last one
    if isinstance(y, (list, tuple)):
        y = y[-1]

    # Ensure tensor
    y = tf.convert_to_tensor(y)

    # Normalize to [batch, classes]
    if y.shape.rank == 1:
        y = tf.reshape(y, [1, -1])

    # Binary (sigmoid) vs. multi-class (softmax)
    if y.shape[-1] == 1:
        return int(tf.squeeze(y, axis=-1)[0] >= 0.5)
    else:
        return int(tf.argmax(y[0]))


def cam_percentile_bands(cam):

    cam = np.clip(cam, 0, 1)
    p70 = np.percentile(cam, 70)
    p80 = np.percentile(cam, 80)
    p90 = np.percentile(cam, 90)

    # Keep the original CAM values inside each band.
    band_70_80 = np.where((cam >= p70) & (cam < p80), cam, 0)
    band_80_90 = np.where((cam >= p80) & (cam < p90), cam, 0)
    band_90_100 = np.where(cam >= p90, cam,  0)
    return band_70_80, band_80_90, band_90_100


def combine_heatmaps(heatmap_70_80,
                     heatmap_80_90,
                     heatmap_90_100):

    lower = Image.fromarray(heatmap_70_80, mode="RGBA")
    middle = Image.fromarray(heatmap_80_90, mode="RGBA")
    upper = Image.fromarray(heatmap_90_100, mode="RGBA")

    combined = Image.alpha_composite(lower, middle)
    combined = Image.alpha_composite(combined, upper)

    return np.array(combined)


def run_layercam(model, image_path, img_size=(224, 224),
                 layer_names=("conv3_block4_out", "conv4_block23_out", "conv5_block3_out")):
    try:
        base_image, image_array = load_image(image_path)
        print(f"\nProcessing: {image_path}")
        print(f"Resized image: {base_image.size}")
        print(f"CNN input: {image_array.shape}")

        # Decide which class to explain
        class_index = predict_class_index(model, image_array)
        print(f"Class index: {class_index}")

        cam = layercam_multi(model, image_array, class_index, layer_names, img_size)

        # using the CAM weights and the color scheme, create the CAM only image.
        band_70_80, band_80_90, band_90_100 = cam_percentile_bands(cam)
        heatmap_70_80_array = to_heatmap(band_70_80, cmap=blue, alpha_scale=0.7)
        heatmap_80_90_array = to_heatmap(band_80_90, cmap=purple, alpha_scale=0.7)
        heatmap_90_100_array = to_heatmap(band_90_100, cmap=pink, alpha_scale=0.7)

        # Combine the colored CAMs.
        combined_heatmap_array = combine_heatmaps(heatmap_70_80_array, heatmap_80_90_array, heatmap_90_100_array)
        combined_heatmap_image = Image.fromarray(combined_heatmap_array)

        # Overlay the combined CAM on the original image.
        overlay = overlay_heatmap(base_image, combined_heatmap_array)
        print(f"SUCCESS: {image_path}")

        return combined_heatmap_image, combined_heatmap_array, overlay, class_index

    except Exception:
        print(f"FAILED: {image_path}")
        traceback.print_exc()
        raise


def camoverlays_driver(base_image_dir,
                       original_images_dir,
                       original_images_224_dir,
                       cam_images_dir,
                       cam_overlay_images_dir,
                       image_size):

    try:
        # 0) Set up the folder paths.
        print("0")
        base_dir = base_image_dir
        orig_dir = original_images_dir          # base_dir / "images_testing"
        orig_224_dir = original_images_224_dir  # base_dir / "images_orig224"
        just_cam_dir = cam_images_dir           # base_dir / "images_just_CAM"
        dst_dir = cam_overlay_images_dir        # base_dir / "images_overlay_CAM"
        img_size = image_size
        print("1")

        # 1) Generate images from the originals to dimensions used by the model during training.
        generate_images_orig224(source_folder=orig_dir, output_folder=orig_224_dir)
        print("2")

        # 2) Create a ResNet101 model and load the weights.
        _, model = get_model(weights=None)  # get the model, skip the backbone 1st parameter.
        weights_folder = base_dir / "images_testing/GAweights.h5"
        model.load_weights(weights_folder, by_name=True, skip_mismatch=False)
        print("3")

        # 3) Select a layer from the model.
        # LAYERS = ["conv3_block4_out"]      # final CAM is 28x28; grid of dots
        # LAYERS = ["conv4_block23_out"]     # final CAM is 14x14; bigger grid of dots
        LAYERS = ["conv5_block2_out"]        # final CAM is 7x7; this worked.
        # LAYERS = ["conv3_block4_out","conv4_block23_out","conv5_block2_out"] # never mind.

        # 4) Loop through the original images (with new dimensions) and create CAM overlays.
        for img_path in sorted(orig_224_dir.iterdir()):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() != ".png":  # make sure that we only process image files.
                continue

            out_name = img_path.stem + ".png"
            out_path_justcam = just_cam_dir / out_name  # the cam image has the same name as the original, but different directory.
            out_path_camoverlay = dst_dir / out_name    # the cam overlay has the same name as the original, but different directory.
            print("inside CAMintermediateLayer ", img_path)

            cam_image, cam_array, cam_overlay_image, explained_class = run_layercam(model,
                                                                                    img_path,
                                                                                    img_size=img_size,
                                                                                    layer_names=LAYERS)

            # Save the CAM image and the CAM overlay image; copy over existing files.
            cam_image.save(out_path_justcam)
            cam_overlay_image.save(out_path_camoverlay)

        print("Leaving GA_camoverlays.py ")

    except Exception as e:
        print(f"An error occurred in GA_camoverlays.camoverlays_driver: {e}")



