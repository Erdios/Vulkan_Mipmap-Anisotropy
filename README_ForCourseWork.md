# HPG Coursework 01
Coursework from HPG Programme in University of Leeds.

# Changes
Code are basically from Vulkan Exercises.

Exercise 1.1 - 3D Scene

Exercise 1.2 - 3D Scene
1. New structure _ModelBufferPack_ in **vertex_data.h** and **vertex_data.cpp**. This structure group objects that belong to a sub mesh from a mesh mesh.
2. Pass a group of instances from _ModelBufferPack_ structure to the function _record_commands_ in **main.cpp**.
3. new function _create_image_texture2d_with_solid_color()_ for the structure in **vkimage.h** and **vkimage.cpp**.

Exercise 1.3 User camera
1.  New classes for camera and mouse control in **camera_control.h** and **camera_control.cpp**.
2.  Add new GLFW window events in **main.cpp** to listen to the keyboard input and mouse input.

Exercise 1.4 Mipmap
1. Code for generating mipmap is in the function _load_image_texture2d_with_bliting_ function in **vkimage.hpp** and **vkimage.cpp**. They were written based on an online [Vulkan tutorial](https://vulkan-tutorial.com/Generating_Mipmaps).

Exercise 1.5 Anisotropy
1. Only little changes for anisotropy when creating physical device and sampler in **vulkan_window.hpp**, **vulkan_window.cpp**, **vertex_data.hpp** and **vertex_data.cpp**. Based on [tutorial](https://vulkan-tutorial.com/Texture_mapping/Image_view_and_sampler#page_Anisotropy-device-feature)

    Without Anisotropy:

    ![without_anisotropy](./PNGs/without_anisotropy.png)

    Without Anisotropy:

    ![with_anisotropy](./PNGs/with_anisotropy.png)

    


# Third party lib
Only use the third party lib from Vulkan Exercises.
