# The Georgia project on https://github.com/KatherineMossDeveloper/The-Georgia-Project/tree/main
# GApredict.py
#
# This code will pull image files from a folder and do inference on each one, reporting the
# classification and confidence to the output window.  The prediction logic below assumes
# a binary classification, where the CEX images are in folder '0' and PG are in folder '1'.
# By convention, '1' is the positive class in binary classification, meaning the class that
# the model is trained to predict.
#
# The parameter file_type is the suffix of the image file, in this case ".png" is the default.
#
# The parameter mod is short for modulo.  The modulo operator "%" give you the remainder
# after dividing.  Because the default value for mod is 1, every image will be processed
# in the loop.  If you want to debug something with fewer images, change the mod to a
# higher number, so that fewer images will be processed.
#
# Note that the model in GAmodel is resnet101, but the preprocess_input is from resnet50
# because the pre-processing is identical, and keras did not create one for the resnet101.
#
# To do.
# (nothing)
# #############################################################################################

import os
from GAutility import load_and_preprocess_image


# Inference for each image in the folder
def predict_driver(model, image_folder, file_type = ".png", mod = 1):
    confidence_list = []
    file_paths = []
    counter = 0

    try:
        # Load the pre-trained model with weights
        print(f'Starting GApredict.py.')

        for filename in sorted(os.listdir(image_folder)):
            counter = counter + 1

            if filename.endswith(file_type) and counter % mod == 0:
                file_path = os.path.join(image_folder, filename)
                file_paths.append(file_path)

                # Load and preprocess the image
                img_array = load_and_preprocess_image(file_path)

                # Perform inference on the image
                prediction = model.predict(img_array, verbose=0 )

                # Assuming binary classification folder structure CEX (0) & PG (1),
                # this will return the confidence that the image is of PG.
                confidence = prediction[0][0]

                # Convert to percentage
                confidence_percent = confidence * 100
                # convert % when CEX...confidence_adjusted = confidence_percent if confidence_percent >= 50 else (100 - confidence_percent)
                confidence_list.append(int(round(confidence_percent)))

                # Assign the class label with threshold 0.5 for PG.
                class_label = 'PG' if confidence >= 0.5 else 'CEX'  # PG if prob >= 0.5, else CEX

                # Print the result with confidence as a percentage
                print(f"File {filename} prediction: {class_label} with confidence {confidence_percent:.2f}%")

    except Exception as e:
        print(f"An error occurred in GApredict: {e}")

    print(f'counter {counter}')

    return confidence_list, file_paths

