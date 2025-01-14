# Creating Visual Cognitive Illusions
---
Welcome to our repository [Firefly2024/Diffusion-Illusions](https://github.com/Firefly2024/Diffusion-Illusions)

This repository contains the implementation of **Creating Visual Cognitive Illusions**,which is the final project of Computer Vision,2024 fall,Peking University.

Worked by 连昊 张煦恒 张江宇

This project is based on the open-source project **Diffusion-Illusions**, grateful to them!

Their website:[diffusionillusions.com](https://diffusionillusions.com)

Their github repository:[ryanndagreat.github.io/Diffusion-Illusions](https://ryanndagreat.github.io/Diffusion-Illusions)
 
- [Creating Visual Cognitive Illusions](#creating-visual-cognitive-illusions)
  - [1.Project structure](#1project-structure)
  - [2.How to run the code](#2how-to-run-the-code)
 
## 1.Project structure
---
This repository retains all files from the original project, and any files that we have modified or added have been renamed accordingly. 

The main runnable Jupyter Notebook:

    more_shapes_puzzle_colab.ipynb: The implementation of more complex arangement operations(Klotski and Tangram)
    image_changer_colab.ipynb: The implementation of flexible image shapes(triangle and circle)
    kaleidoscope_for_colab.ipynb: The implementation of kaleidoscope effect
    hidden_characters_with_images.ipynb: The implementation of generating hidden characters images given image prompts
    hybrid_image_colab.ipynb: The implementation of generating hybrid images given a pair of text prompts
    hybrid_extension_colab.ipynb: The implementation of generating extended hybrid image given a text prompt and a control image

The source python file:

    source/uv_map_generator.py: The implementation of UV_MAP generator
    source/stable_diffusion_with_images.py: The implementation of stable diffusion (especially train_step) working for image prompts

## 2.How to run the code
---
The code has been organised as a Jupyter Notebook and has been tested on Google Colab.Therefore, we recommend running this code on Google Colab.
All the environmental preparation and model loader has been set up in every notebook, so you only need to follow the instructions in every notebook.