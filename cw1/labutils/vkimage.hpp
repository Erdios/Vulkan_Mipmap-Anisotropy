#pragma once

#include <volk/volk.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>
#include <utility>

#include <cassert>

#include "allocator.hpp"

namespace labutils
{
	class Image
	{
		public:
			Image() noexcept, ~Image();

			explicit Image( VmaAllocator, VkImage = VK_NULL_HANDLE, VmaAllocation = VK_NULL_HANDLE ) noexcept;

			Image( Image const& ) = delete;
			Image& operator= (Image const&) = delete;

			Image( Image&& ) noexcept;
			Image& operator = (Image&&) noexcept;

		public:
			VkImage image = VK_NULL_HANDLE;
			VmaAllocation allocation = VK_NULL_HANDLE;

		private:
			VmaAllocator mAllocator = VK_NULL_HANDLE;
	};

	Image load_image_texture2d(char const* aPattern, VulkanContext const&, VkCommandPool, Allocator const&);
	Image load_image_texture2d_with_bliting(char const* aPattern, VulkanContext const&, VkCommandPool, Allocator const&);
	Image load_image_texture2d_no_minmap(char const* aPattern, VulkanContext const& aContext, VkCommandPool aCmdPool, Allocator const& aAllocator);

	Image create_image_texture2d_with_solid_color(char const* aPattern, VulkanContext const& aContext, VkCommandPool aCmdPool, Allocator const& aAllocator, glm::vec4 inColor);

	Image create_image_texture2d( Allocator const&, std::uint32_t aWidth, std::uint32_t aHeight, VkFormat, VkImageUsageFlags = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, std::uint32_t mipLevels = 1);

	std::uint32_t compute_mip_level_count( std::uint32_t aWidth, std::uint32_t aHeight );
}
