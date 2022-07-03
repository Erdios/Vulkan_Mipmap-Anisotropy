#version 450

// inputs
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inTexcoord;

// outputs
layout( location = 0 ) out vec2 v2fTexCoord;

// uniform
layout(set = 0, binding = 0) uniform UScene
{
	mat4 camera;
	mat4 projection;
	mat4 projCam;
}uScene;

void main()
{
	v2fTexCoord = inTexcoord;
	gl_Position = uScene.projCam * vec4( inPosition.xyz, 1.f ); 
}