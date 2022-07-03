#include <volk/volk.h>

#include <tuple>
#include <chrono>
#include <limits>
#include <vector>
#include <iostream>
#include <stdexcept>

#include <cstdio>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#if !defined(GLM_FORCE_RADIANS)
#	define GLM_FORCE_RADIANS
#endif
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "camera_control.h"
#include "../labutils/to_string.hpp"
#include "../labutils/vulkan_window.hpp"

#include "../labutils/angle.hpp"
using namespace labutils::literals;

#include "../labutils/error.hpp"
#include "../labutils/vkutil.hpp"
#include "../labutils/vkimage.hpp"
#include "../labutils/vkobject.hpp"
#include "../labutils/vkbuffer.hpp"
#include "../labutils/allocator.hpp"
#include "vertex_data.h"
namespace lut = labutils;

#include "model.hpp"

namespace
{

	namespace cfg
	{
		// Compiled shader code for the graphics pipeline(s)
		// See sources in cw1/shaders/*. 
#		define SHADERDIR_ "assets/cw1/shaders/"
		constexpr char const* kVertShaderPath = SHADERDIR_ "default.vert.spv";
		constexpr char const* kFragShaderPath = SHADERDIR_ "default.frag.spv";
#		undef SHADERDIR_

		
		
#		define SCENE_ "assets/cw1/scenes/"
		constexpr char const* carObjectPath = SCENE_ "car.obj";
		constexpr char const* cityObjectPath = SCENE_ "city.obj";
#		undef SCENE_


		// General rule: with a standard 24 bit or 32 bit float depth buffer,
		// you can support a 1:1000 ratio between the near and far plane with
		// minimal depth fighting. Larger ratios will introduce more depth
		// fighting problems; smaller ratios will increase the depth buffer's
		// resolution but will also limit the view distance.
		constexpr float kCameraNear  = 0.1f;
		constexpr float kCameraFar   = 10000.f;
		constexpr auto kCameraFov    = 60.0_degf;

		constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;
	}



	// Uniform data
	namespace glsl
	{
		
		ControlComponent::Camera camera;
		ControlComponent::Mouse mouse;
		
		struct SceneUniform
		{
			glm::mat4 camera;
			glm::mat4 projection;
			glm::mat4 projCam;
		};

		static_assert(sizeof(SceneUniform) <= 65536, "SceneUniform must be less than 65536 bytes for vkCmdUpdateBuffer.");
		static_assert(sizeof(SceneUniform) % 4 == 0, "SceneUniform size must be a multiple of 4 bytes.");
	}


	// Local types/structures:

	// Local functions:
	lut::RenderPass create_render_pass(lut::VulkanWindow const& );
	lut::DescriptorSetLayout create_descriptor_layout(lut::VulkanWindow const& aWindow, VkDescriptorType, VkShaderStageFlags);
	lut::PipelineLayout create_pipeline_layout(lut::VulkanContext const&);
	lut::PipelineLayout create_pipeline_layout(lut::VulkanContext const& aContext, std::vector<VkDescriptorSetLayout> vaSceneLayouts);
	lut::Pipeline create_pipeline(lut::VulkanWindow const& , VkRenderPass , VkPipelineLayout );
	
	
	void create_swapchain_framebuffers(lut::VulkanWindow const& , VkRenderPass , std::vector<lut::Framebuffer>&, VkImageView aDepthView);
	void record_commands( VkCommandBuffer, VkRenderPass, VkFramebuffer, VkPipeline, VkPipelineLayout, VkExtent2D const&, 
		std::vector<ModelBufferPack>&,VkBuffer uniformBuffer, VkDescriptorSet matrixDescriptorSet, glsl::SceneUniform matrixUniform);
	void submit_commands( lut::VulkanContext const&, VkCommandBuffer, VkFence, VkSemaphore, VkSemaphore);
	void update_scene_uniforms(glsl::SceneUniform& aSceneUniforms, std::uint32_t aFramebufferWidth, std::uint32_t aFramebufferHeight);
	std::tuple<lut::Image, lut::ImageView> create_depth_buffer(lut::VulkanWindow const& aWindow, lut::Allocator const& aAllocator);
	void update_descriptor_set(lut::VulkanWindow const& window, VkBuffer descriptorBuffer, VkDescriptorSet descritporSet, VkDescriptorType descriptorType);
}

// Definitions of functions
namespace
{

	// window & camera control
	void glfw_callback_key_press(GLFWwindow* aWindow, int aKey, int /*aScanCode*/, int aAction, int /*aModifierFlags*/)
	{
		
		if (GLFW_PRESS == aAction)
		{	
			// close window
			if (GLFW_KEY_ESCAPE == aKey)
				glfwSetWindowShouldClose(aWindow, GLFW_TRUE);
			// adjust speed
			else if (GLFW_KEY_LEFT_SHIFT == aKey || GLFW_KEY_RIGHT_SHIFT == aKey)
				glsl::camera.speedChangeMode = ControlComponent::Camera::SpeedUp;
			else if (GLFW_KEY_LEFT_CONTROL == aKey || GLFW_KEY_RIGHT_CONTROL == aKey)
				glsl::camera.speedChangeMode = ControlComponent::Camera::SpeedDown;
			// translation
			else if(aKey == GLFW_KEY_Q)
				glsl::camera.ifKeyQPressed = true;
			else if (aKey == GLFW_KEY_W)
				glsl::camera.ifKeyWPressed = true;
			else if (aKey == GLFW_KEY_E)
				glsl::camera.ifKeyEPressed = true;
			else if (aKey == GLFW_KEY_A)
				glsl::camera.ifKeyAPressed = true;
			else if (aKey == GLFW_KEY_S)
				glsl::camera.ifKeySPressed = true;
			else if (aKey == GLFW_KEY_D)
				glsl::camera.ifKeyDPressed = true;
		}

		if (GLFW_RELEASE == aAction)
		{
			// don't adjust speed
			if (GLFW_KEY_LEFT_CONTROL == aKey || GLFW_KEY_LEFT_SHIFT == aKey || GLFW_KEY_RIGHT_CONTROL == aKey || GLFW_KEY_RIGHT_SHIFT == aKey)
			{
				glsl::camera.speedChangeMode = ControlComponent::Camera::NoChange;
			}
			else if (aKey == GLFW_KEY_Q)
				glsl::camera.ifKeyQPressed = false;
			else if (aKey == GLFW_KEY_W)
				glsl::camera.ifKeyWPressed = false;
			else if (aKey == GLFW_KEY_E)
				glsl::camera.ifKeyEPressed = false;
			else if (aKey == GLFW_KEY_A)
				glsl::camera.ifKeyAPressed = false;
			else if (aKey == GLFW_KEY_S)
				glsl::camera.ifKeySPressed = false;
			else if (aKey == GLFW_KEY_D)
				glsl::camera.ifKeyDPressed = false;
		}
	}

	static void mouse_pos_callback(GLFWwindow* window, double xpos, double ypos)
	{
		glsl::mouse.currentPos.x = xpos;
		glsl::mouse.currentPos.y = ypos;


		if (glsl::mouse.isActivated == true)
		{
			glsl::camera.rotate_camera(glsl::mouse.currentPos - glsl::mouse.previousPos);
			glsl::mouse.previousPos = glsl::mouse.currentPos;
		}

	}

	static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
	{
		
		if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE)
		{
			glsl::mouse.isActivated = !glsl::mouse.isActivated;
			glsl::mouse.previousPos = glsl::mouse.currentPos;
		}
		
	}


	// rendering preparation
	lut::RenderPass create_render_pass(lut::VulkanWindow const& aWindow)
	{
		//------------//
		// Attachment //
		//------------//

		VkAttachmentDescription attachments[2]{}; // ONLY ONE attachment

		// For attachment 0
		attachments[0].format = aWindow.swapchainFormat; // VK FORMAT R8G8B8A8 SRGB 
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT; // no multisampling 
		// load and store operations
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		// layout
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		// For attachment 1
		attachments[1].format = cfg::kDepthFormat;
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;


		//------------//
		// Subpass    //
		//------------//

		// create attachment reference
		VkAttachmentReference colorAttachments[1]{};
		colorAttachments[0].attachment = 0; // the zero refers to attachments[0] declared earlier.
		//specify the layout for attachment image
		colorAttachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		// create depth buffer attachment reference
		VkAttachmentReference depthAttachment{};
		depthAttachment.attachment = 1;
		depthAttachment.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		// create subpass
		VkSubpassDescription subpasses[1]{};
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[0].colorAttachmentCount = 1; // one attachment
		subpasses[0].pColorAttachments = colorAttachments;
		subpasses[0].pDepthStencilAttachment = &depthAttachment;


		//-------------------------//
		// Create render pass      //
		//-------------------------//

		VkRenderPassCreateInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passInfo.attachmentCount = 2;
		passInfo.pAttachments = attachments;
		passInfo.subpassCount = 1;
		passInfo.pSubpasses = subpasses;
		passInfo.dependencyCount = 0;
		passInfo.pDependencies = nullptr;

		VkRenderPass rpass = VK_NULL_HANDLE;
		if (auto const res = vkCreateRenderPass(aWindow.device, &passInfo, nullptr, &rpass); VK_SUCCESS != res)
		{

			throw lut::Error("Unable to create render pass\n"
				"vkCreateRenderPass() returned %s", lut::to_string(res).c_str()
			);

		}

		return lut::RenderPass(aWindow.device, rpass);
	}

	lut::DescriptorSetLayout create_descriptor_layout(lut::VulkanWindow const& aWindow, VkDescriptorType descriptorType, VkShaderStageFlags shaderStageFlag)
	{
		//1. Define the descriptor set layout
		VkDescriptorSetLayoutBinding bindings[1]{};
		//match the binding id in shaders
		bindings[0].binding = 0;
		bindings[0].descriptorType = descriptorType;
		bindings[0].descriptorCount = 1;
		//specifying which pipeline shader stages can access a resource for this binding.
		bindings[0].stageFlags = shaderStageFlag;

		//2. Create the descriptor set layout
		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;

		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create descriptor set layout\n"
				"vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());
		}

		return lut::DescriptorSetLayout(aWindow.device, layout);
	}
	
	lut::PipelineLayout create_pipeline_layout(lut::VulkanContext const& aContext)
	{

		// create pipeline layout information

		VkPipelineLayoutCreateInfo layoutInfo{};

		layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

		layoutInfo.setLayoutCount = 0;

		layoutInfo.pSetLayouts = nullptr;

		layoutInfo.pushConstantRangeCount = 0;

		layoutInfo.pPushConstantRanges = nullptr;


		// create pipeline layout

		VkPipelineLayout layout = VK_NULL_HANDLE;

		if (auto const res = vkCreatePipelineLayout(aContext.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{

			throw lut::Error("Unable to create pipeline layout\n"
				"vkCreatePipelineLayout() returned %s", lut::to_string(res).c_str()
			);
		}

		return lut::PipelineLayout(aContext.device, layout);
	}
	
	lut::PipelineLayout create_pipeline_layout(lut::VulkanContext const& aContext, std::vector<VkDescriptorSetLayout> vaSceneLayouts)
	{
		//VkDescriptorSetLayout layouts[] = { vaSceneLayout, aObjectLayout };

		// create pipeline layout information
		VkPipelineLayoutCreateInfo layoutInfo{};

		layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutInfo.setLayoutCount = vaSceneLayouts.size();
		layoutInfo.pSetLayouts = vaSceneLayouts.data();
		layoutInfo.pushConstantRangeCount = 0;
		layoutInfo.pPushConstantRanges = nullptr;


		// create pipeline layout

		VkPipelineLayout layout = VK_NULL_HANDLE;

		if (auto const res = vkCreatePipelineLayout(aContext.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{

			throw lut::Error("Unable to create pipeline layout\n"
				"vkCreatePipelineLayout() returned %s", lut::to_string(res).c_str()
			);
		}

		return lut::PipelineLayout(aContext.device, layout);
	}
	
	lut::Pipeline create_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout)
	{
		// load shader modules
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::kVertShaderPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::kFragShaderPath);


		// create pipeline shader stage instance
		VkPipelineShaderStageCreateInfo stages[2]{}; // for vert shader and frag shader respectively


		// stage for vertex shader
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		// stage for fragment shader
		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";

		// vertex input
		VkVertexInputBindingDescription vertexInputs[2]{};
		vertexInputs[0].binding = 0;
		vertexInputs[0].stride = sizeof(float) * 3;
		vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		vertexInputs[1].binding = 1;
		vertexInputs[1].stride = sizeof(float) * 2;
		vertexInputs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		// vertex attributes
		VkVertexInputAttributeDescription vertexAttributes[2]{};
		vertexAttributes[0].binding = 0; // must match binding above
		vertexAttributes[0].location = 0; // must match shader
		vertexAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[0].offset = 0;

		vertexAttributes[1].binding = 1; // must match binding above
		vertexAttributes[1].location = 1; // must match shader
		vertexAttributes[1].format = VK_FORMAT_R32G32_SFLOAT;
		vertexAttributes[1].offset = 0;

		// define primitive of input
		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		// Define viewport and scissor regions
		VkViewport viewport{};
		viewport.x = 0.f;
		viewport.y = 0.f;
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.f;
		viewport.maxDepth = 1.f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0, 0 };
		scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width, aWindow.swapchainExtent.height };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		// vertex input info
		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.vertexBindingDescriptionCount = 2;
		inputInfo.pVertexBindingDescriptions = vertexInputs;
		inputInfo.vertexAttributeDescriptionCount = 2;
		inputInfo.pVertexAttributeDescriptions = vertexAttributes;
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;


		// depth stencil state create info
		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.f;
		depthInfo.maxDepthBounds = 1.f;


		// Define rasterization options
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.depthBiasEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.lineWidth = 1.f; // required. 

		// Define multisampling state
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT; // only one sample per pixel

		// Define blend state
		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorWriteMask =
			VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;

		// Create pipeline
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

		pipeInfo.stageCount = 2; // vertex + fragment stages
		pipeInfo.pStages = stages;

		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr; // no tessellation 
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = &depthInfo; // no depth or stencil buffers 
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = nullptr; // no dynamic states 

		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0; // first subpass of aRenderPass 

		VkPipeline pipe = VK_NULL_HANDLE;
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); res != VK_SUCCESS)
		{

			throw lut::Error("Unable to create graphics pipeline\n"
				"vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str());

		}

		return lut::Pipeline(aWindow.device, pipe);
	}

	void create_swapchain_framebuffers(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, std::vector<lut::Framebuffer>& aFramebuffers, VkImageView aDepthView)
	{
		assert(aFramebuffers.empty());

		for (std::uint32_t i = 0; i < aWindow.swapViews.size(); i++)
		{

			VkImageView attachments[2] = { aWindow.swapViews[i], aDepthView };

			VkFramebufferCreateInfo fbInfo{};

			fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			fbInfo.flags = 0;
			fbInfo.renderPass = aRenderPass;
			fbInfo.attachmentCount = 2; // two attachment for one image
			fbInfo.pAttachments = attachments;
			fbInfo.width = aWindow.swapchainExtent.width;
			fbInfo.height = aWindow.swapchainExtent.height;
			fbInfo.layers = 1;

			VkFramebuffer fb = VK_NULL_HANDLE;

			if (auto const res = vkCreateFramebuffer(aWindow.device, &fbInfo, nullptr, &fb); res != VK_SUCCESS)
			{
				throw lut::Error(
					"Unable to create framebuffer for swap chain image %zu\n"
					"vkCreateFramebuffer() returned %s", i, lut::to_string(res).c_str()
				);

			}


			aFramebuffers.emplace_back(lut::Framebuffer(aWindow.device, fb));

		}

		assert(aWindow.swapViews.size() == aFramebuffers.size());
	}

	void update_scene_uniforms(glsl::SceneUniform& aSceneUniforms, std::uint32_t aFramebufferWidth, std::uint32_t aFramebufferHeight)
	{
		//TODO- (Section 3) initilize SceneUniform members

		// aspect for framebuffer
		float const aspect = aFramebufferWidth / float(aFramebufferHeight);

		// get projection matrix
		aSceneUniforms.projection = glm::perspectiveRH_ZO(
			lut::Radians(cfg::kCameraFov).value(),
			aspect,
			cfg::kCameraNear,
			cfg::kCameraFar
		);

		aSceneUniforms.projection[1][1] *= -1.f; // mirror Y axis

		aSceneUniforms.camera = glsl::camera.get_view_matrix();

		aSceneUniforms.projCam = aSceneUniforms.projection * aSceneUniforms.camera;
	}
	
	std::tuple<lut::Image, lut::ImageView> create_depth_buffer(lut::VulkanWindow const& aWindow, lut::Allocator const& aAllocator)
	{
		VkImageCreateInfo imageInfo{};

		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = cfg::kDepthFormat;
		imageInfo.extent.width = aWindow.swapchainExtent.width;
		imageInfo.extent.height = aWindow.swapchainExtent.height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

		VkImage image = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;

		if (auto const res = vmaCreateImage(aAllocator.allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr); res != VK_SUCCESS)
		{
			throw lut::Error("Unable to allocate depth buffer image.\nvmaCreateImage() returned %s", lut::to_string(res).c_str());
		}

		lut::Image depthImage(aAllocator.allocator, image, allocation);

		// image view
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = depthImage.image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = cfg::kDepthFormat;
		viewInfo.components = VkComponentMapping{};
		viewInfo.subresourceRange = VkImageSubresourceRange{
			VK_IMAGE_ASPECT_DEPTH_BIT,
			0,1,
			0,1
		};

		VkImageView view = VK_NULL_HANDLE;
		if (auto const res = vkCreateImageView(aWindow.device, &viewInfo, nullptr, &view); res != VK_SUCCESS)
		{
			throw lut::Error("Unable to create image view\nvkCreateImageView() returned %s", lut::to_string(res).c_str());
		}

		return { std::move(depthImage), lut::ImageView{aWindow.device, view} };
	}
	
	void update_descriptor_set(lut::VulkanWindow const& window, VkBuffer descriptorBuffer, VkDescriptorSet descritporSet, VkDescriptorType descriptorType)
	{
		// Write descriptor set
		VkWriteDescriptorSet desc[1]{};

		// Descriptor Buffer Info 
		VkDescriptorBufferInfo descBufferInfo{};
		descBufferInfo.buffer = descriptorBuffer;
		descBufferInfo.range = VK_WHOLE_SIZE;

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = descritporSet;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = descriptorType;
		desc[0].descriptorCount = 1;
		desc[0].pBufferInfo = &descBufferInfo;

		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}
	
	// run cmd commands
	void record_commands(VkCommandBuffer aCmdBuff, VkRenderPass aRenderPass, VkFramebuffer aFramebuffer, VkPipeline aGraphicsPipe, VkPipelineLayout aGraphicsPipeLayout,
		VkExtent2D const& aImageExtent, std::vector<ModelBufferPack>& mesh, VkBuffer matrixUBO, VkDescriptorSet matrixDescriptorSet, glsl::SceneUniform matrixUniform)
	{

		// Begin recording commands
		VkCommandBufferBeginInfo begInfo{};
		begInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		begInfo.pInheritanceInfo = nullptr;

		if (auto const res = vkBeginCommandBuffer(aCmdBuff, &begInfo); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to begin recording command buffer\n"
				"vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
		}


		// Begin render pass
		VkClearValue clearValues[2]{};
		clearValues[0].color.float32[0] = 0.1f; // Clear to a dark gray background. 
		clearValues[0].color.float32[1] = 0.1f; // If we were debugging, this would potentially 
		clearValues[0].color.float32[2] = 0.1f; // help us see whether the render pass took 
		clearValues[0].color.float32[3] = 1.f;  // place, even if nothing else was drawn.

		clearValues[1].depthStencil.depth = 1.f;

		// Barrier One
		lut::buffer_barrier(
			aCmdBuff,
			matrixUBO,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT
		);

		// Update uniform buffer
		vkCmdUpdateBuffer(aCmdBuff, matrixUBO, 0, sizeof(glsl::SceneUniform), &matrixUniform);

		// Barrier Two
		lut::buffer_barrier(
			aCmdBuff,
			matrixUBO,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT
		);

		VkRenderPassBeginInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		passInfo.renderPass = aRenderPass;
		passInfo.framebuffer = aFramebuffer;
		passInfo.renderArea.offset = VkOffset2D{ 0, 0 };
		passInfo.renderArea.extent = VkExtent2D{ aImageExtent.width, aImageExtent.height };
		passInfo.clearValueCount = 2;
		passInfo.pClearValues = clearValues;
		vkCmdBeginRenderPass(aCmdBuff, &passInfo, VK_SUBPASS_CONTENTS_INLINE);


		// Begin drawing with our graphics pipeline
		vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsPipe);
		
		for (unsigned int i = 0; i < mesh.size(); ++i)
		{
			// Binding vertex buffers
			VkBuffer buffers[2] = { mesh[i].positions.buffer, mesh[i].texcoords.buffer};
			VkDeviceSize offsets[2]{};
			vkCmdBindVertexBuffers(aCmdBuff, 0, 2, buffers, offsets);

			// Binding descriptor sets
			vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsPipeLayout, 0, 1, &matrixDescriptorSet, 0, nullptr);
			vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsPipeLayout, 1, 1, &mesh[i].materialDescriptorSet, 0, nullptr);

			// Draw a mesh
			vkCmdDraw(aCmdBuff, mesh[i].vertexCount, 1, 0, 0);
		}

		// End the render pass 
		vkCmdEndRenderPass(aCmdBuff);

		// End command recording
		if (auto const res = vkEndCommandBuffer(aCmdBuff); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to end recording command buffer\n"
				"vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
		}
	}

	void submit_commands(lut::VulkanContext const& aContext, VkCommandBuffer aCmdBuff, VkFence aFence, VkSemaphore aWaitSemaphore, VkSemaphore aSignalSemaphore)
	{
		VkPipelineStageFlags waitPipelineStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &aCmdBuff;


		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = &aWaitSemaphore;
		submitInfo.pWaitDstStageMask = &waitPipelineStages;

		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &aSignalSemaphore;


		if (auto const res = vkQueueSubmit(aContext.graphicsQueue, 1, &submitInfo, aFence);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to submit command buffer to queue\n"
				"vkQueueSubmit() returned %s", lut::to_string(res).c_str());
		}
	}
}

int main() try
{
	//DOING-implement me.
	
	// Create Vulkan Window
	lut::VulkanWindow window = lut::make_vulkan_window();
	// Configure the GLFW window
	glfwSetKeyCallback(window.window, &glfw_callback_key_press);
	glfwSetCursorPosCallback(window.window, &mouse_pos_callback);
	glfwSetMouseButtonCallback(window.window, &mouse_button_callback);

	// Create VMA allocator
	lut::Allocator allocator = lut::create_allocator(window);
	
	// Render pass
	lut::RenderPass renderPass = create_render_pass(window);

	// Create descriptor set layout
	lut::DescriptorSetLayout matrixLayout = create_descriptor_layout(window, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT);
	lut::DescriptorSetLayout materialLayout = create_descriptor_layout(window, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT);

	// Pipeline
	lut::PipelineLayout pipeLayout = create_pipeline_layout(window, { matrixLayout.handle, materialLayout.handle });
	lut::Pipeline pipe = create_pipeline(window, renderPass.handle, pipeLayout.handle);

	// Create depth buffer
	auto [depthBuffer, depthBufferView] = create_depth_buffer(window, allocator);

	// Framebuffer
	std::vector<lut::Framebuffer> framebuffers;
	create_swapchain_framebuffers(window, renderPass.handle, framebuffers, depthBufferView.handle);

	// Command
	lut::CommandPool cpool = lut::create_command_pool(window, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
	std::vector<VkCommandBuffer> cbuffers;
	std::vector<lut::Fence> cbfences;
	
	for (std::size_t i = 0; i < framebuffers.size(); ++i)
	{
		cbuffers.emplace_back(lut::alloc_command_buffer(window, cpool.handle));
		cbfences.emplace_back(lut::create_fence(window, VK_FENCE_CREATE_SIGNALED_BIT));
	}

	// Semaphore
	lut::Semaphore imageAvailable = lut::create_semaphore(window);
	lut::Semaphore renderFinished = lut::create_semaphore(window);

	
	
	// create scene uniform buffer with lut::create_buffer()
	// Create buffer for uniform
	lut::Buffer matrixUBO = lut::create_buffer(
		allocator,
		sizeof(glsl::SceneUniform),
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY
	);

	// create descriptor pool
	lut::DescriptorPool dpool = lut::create_descriptor_pool(window);

	// allocate descriptor set for uniform buffer
	VkDescriptorSet matrixDescriptors = lut::alloc_desc_set(window, dpool.handle, matrixLayout.handle);
	
	update_descriptor_set(window, matrixUBO.buffer, matrixDescriptors, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
	

	// Load mesh
	ModelData cityModel = load_obj_model(cfg::cityObjectPath);
	ModelData carModel = load_obj_model(cfg::carObjectPath);
	std::vector<ModelBufferPack> modelBuffer;

	for(int i =0 ; i<cityModel.meshes.size(); ++i)
		modelBuffer.emplace_back(create_model_buffer_pack(window, allocator, cityModel, materialLayout.handle, dpool.handle, i));

	for (int i = 0; i < carModel.meshes.size(); ++i)
		modelBuffer.emplace_back(create_model_buffer_pack(window, allocator, carModel, materialLayout.handle, dpool.handle, i));

	// Application main loop
	bool recreateSwapchain = false;

	while (!glfwWindowShouldClose(window.window))
	{
		// window event check
		glfwPollEvents(); 

		// Recreate swap chain
		if (recreateSwapchain)
		{
			//re-create swapchain and associated resources!
			vkDeviceWaitIdle(window.device);

			auto const changes = recreate_swapchain(window);

			// re-create render pass
			if (changes.changedFormat)
			{
				renderPass = create_render_pass(window);
			}


			// re-create pipeline
			if (changes.changedSize)
			{
				pipe = create_pipeline(window, renderPass.handle, pipeLayout.handle);
				std::tie(depthBuffer, depthBufferView) = create_depth_buffer(window, allocator);
			}

			// clear framebuffers in the vector and recreate a new vector of framebuffer
			framebuffers.clear();
			create_swapchain_framebuffers(window, renderPass.handle, framebuffers, depthBufferView.handle);

			// disable recreate 
			recreateSwapchain = false;
			continue;
		}


		// acquire swapchain image.
		std::uint32_t imageIndex = 0;
		auto const acquireRes = vkAcquireNextImageKHR(
			window.device,
			window.swapchain,
			std::numeric_limits<std::uint64_t>::max(),
			imageAvailable.handle,
			VK_NULL_HANDLE,
			&imageIndex
		);
		// check info for the swapchain image
		if (VK_SUBOPTIMAL_KHR == acquireRes || VK_ERROR_OUT_OF_DATE_KHR == acquireRes)
		{
			recreateSwapchain = true;
			continue;
		}

		else if (VK_SUCCESS != acquireRes)
		{
			throw lut::Error("Unable to acquire enxt swapchain image\n"
				"vkAcquireNextImageKHR() returned %s", lut::to_string(acquireRes).c_str()
			);
		}

		// make sure commands are not used
		assert(std::size_t(imageIndex) < cbfences.size());

		// wait for the commands are all not be used
		if (auto const res = vkWaitForFences(window.device, 1, &cbfences[imageIndex].handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max());
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to wait for command buffer fence %u\n"
				"vkWaitForFences() returned %s", imageIndex, lut::to_string(res).c_str()
			);
		}

		// reset the fences to be unsignalled
		if (auto const res = vkResetFences(window.device, 1, &cbfences[imageIndex].handle)
			; VK_SUCCESS != res)
		{
			throw lut::Error("Unable to reset command buffer fence %u\n"
				"vkResetFences() returned %s", imageIndex, lut::to_string(res).c_str()
			);
		}


		// Prepare data for this frame
		glsl::SceneUniform matrixUniforms{};
		update_scene_uniforms(matrixUniforms, window.swapchainExtent.width,
			window.swapchainExtent.height);

		assert(std::size_t(imageIndex) < cbuffers.size());
		assert(std::size_t(imageIndex) < framebuffers.size());

		// record and submit commands
		record_commands(
			cbuffers[imageIndex],
			renderPass.handle,
			framebuffers[imageIndex].handle,
			pipe.handle,
			pipeLayout.handle,
			window.swapchainExtent,
			modelBuffer,
			matrixUBO.buffer,
			matrixDescriptors,
			matrixUniforms
		);

		submit_commands(
			window,
			cbuffers[imageIndex],
			cbfences[imageIndex].handle,
			imageAvailable.handle,
			renderFinished.handle
		);


		//TODO: present rendered images.
		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = &renderFinished.handle;
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = &window.swapchain;
		presentInfo.pImageIndices = &imageIndex;
		presentInfo.pResults = nullptr;


		auto const presentRes = vkQueuePresentKHR(window.presentQueue, &presentInfo);
		if (VK_SUBOPTIMAL_KHR == presentRes || VK_ERROR_OUT_OF_DATE_KHR == presentRes)
		{
			recreateSwapchain = true;
		}
		else if (VK_SUCCESS != presentRes)
		{
			throw lut::Error("Unable present swapchain image %u\n"
				"vkQueuePresentKHR() returned %s", imageIndex, lut::to_string(presentRes).c_str()
			);

		}

	}

	// Cleanup takes place automatically in the destructors, but we sill need
	// to ensure that all Vulkan commands have finished before that.
	vkDeviceWaitIdle(window.device);

	return 0;
}
catch( std::exception const& eErr )
{
	std::fprintf( stderr, "\n" );
	std::fprintf( stderr, "Error: %s\n", eErr.what() );
	return 1;
}


//EOF vim:syntax=cpp:foldmethod=marker:ts=4:noexpandtab: 
