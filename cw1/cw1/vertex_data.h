#pragma once

#include "../labutils/vulkan_window.hpp"
#include "../labutils/vkbuffer.hpp"
#include "../labutils/allocator.hpp" 
#include "../cw1/model.hpp"
#include "../labutils/vkutil.hpp"
#include "../labutils/vkimage.hpp"


struct Mesh
{
	labutils::Buffer positions;
	labutils::Buffer texcoords;

	std::string colorTexturePath;
	glm::vec3 color;

	std::uint32_t vertexCount;
};

struct ModelBufferPack
{
	labutils::Buffer positions;
	labutils::Buffer texcoords;
	
	VkDescriptorSetLayout materialSetLayout;
	VkDescriptorSet materialDescriptorSet;
	
	labutils::Image image;
	labutils::ImageView view;
	labutils::Sampler sampler;
	
	std::uint32_t vertexCount;
};


Mesh create_mesh_with_texture(labutils::VulkanContext const&, labutils::Allocator const&, ModelData& const modelData, unsigned int subMeshIndex);


ModelBufferPack create_model_buffer_pack(labutils::VulkanWindow const& window, labutils::Allocator const& allocator,
	ModelData& const modelData, VkDescriptorSetLayout materialSetLayout, VkDescriptorPool dpool, unsigned int subMeshIndex);
