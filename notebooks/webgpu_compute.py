import genstudio.plot as Plot
from genstudio.plot import js

tint = {
    "shader": """
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> uniforms: vec4<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let dims = textureDimensions(inputTex);
  if (gid.x >= dims.x || gid.y >= dims.y) {
    return;
  }
  let srcColor = textureLoad(inputTex, vec2<i32>(gid.xy), 0);
  let tint = vec4<f32>(uniforms.x, uniforms.y, uniforms.z, 1.0);
  let tintedColor = srcColor * tint;
  textureStore(outputTex, vec2<i32>(gid.xy), tintedColor);
}
"""
}

invert = {
    "shader": """
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> uniforms: vec4<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let dims = textureDimensions(inputTex);
  if (gid.x >= dims.x || gid.y >= dims.y) {
    return;
  }
  let srcColor = textureLoad(inputTex, vec2<i32>(gid.xy), 0);
  let invertedColor = vec4<f32>(1.0 - srcColor.r, 1.0 - srcColor.g, 1.0 - srcColor.b, 1.0);
  textureStore(outputTex, vec2<i32>(gid.xy), invertedColor);
}
"""
}

pixelate_by_workgroup = {
    "shader": """
// Texture bindings for input/output and uniform buffer
// uniforms.w contains the pixel block size
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> uniforms: vec4<f32>;

// Each workgroup handles one pixel block
@compute @workgroup_size(1, 1)
fn main(
    @builtin(global_invocation_id) global_id : vec3<u32>,
    @builtin(local_invocation_id) local_id : vec3<u32>,
    @builtin(workgroup_id) group_id : vec3<u32>
) {
    let dims = textureDimensions(inputTex);

    // Extract block size from uniforms
    let blockSize = max(2.0, uniforms.w);

    // Calculate number of blocks needed to cover the image
    let numBlocksX = ceil(f32(dims.x) / blockSize);
    let numBlocksY = ceil(f32(dims.y) / blockSize);

    // Calculate normalized block size to ensure even coverage
    let normalizedBlockSizeX = f32(dims.x) / numBlocksX;
    let normalizedBlockSizeY = f32(dims.y) / numBlocksY;

    // Calculate block coordinates using normalized sizes
    let blockX = u32(round(f32(group_id.x) * normalizedBlockSizeX));
    let blockY = u32(round(f32(group_id.y) * normalizedBlockSizeY));

    // Skip if this block is completely outside texture bounds
    if (blockX >= dims.x || blockY >= dims.y) {
        return;
    }

    // Calculate block boundaries using normalized sizes
    let blockEndX = min(u32(round(f32(group_id.x + 1u) * normalizedBlockSizeX)), dims.x);
    let blockEndY = min(u32(round(f32(group_id.y + 1u) * normalizedBlockSizeY)), dims.y);

    // Calculate average color for this block
    var avgColor = vec4<f32>(0.0);
    var count = 0u;

    // Sum up all pixels in this block
    for (var y = blockY; y < blockEndY; y++) {
        for (var x = blockX; x < blockEndX; x++) {
            avgColor += textureLoad(inputTex, vec2<i32>(i32(x), i32(y)), 0);
            count += 1u;
        }
    }

    // Apply average color to all pixels in this block
    if (count > 0u) {
        avgColor = avgColor / f32(count);

        // Write the averaged color to all pixels in this block
        for (var y = blockY; y < blockEndY; y++) {
            for (var x = blockX; x < blockEndX; x++) {
                // For the first pixel of each block, draw in bright red
                if (x == blockX && y == blockY) {
                    textureStore(outputTex, vec2<i32>(i32(x), i32(y)), vec4<f32>(1.0, 0.0, 0.0, 1.0));
                } else {
                    textureStore(outputTex, vec2<i32>(i32(x), i32(y)), avgColor);
                }
            }
        }
    }
}
""",
    # Create one workgroup per block, using ceil to ensure we cover the entire canvas
    "customDispatch": js(
        "[Math.ceil($state.width / $state.pixelBlockSize), Math.ceil($state.height / $state.pixelBlockSize)]"
    ),
    "workgroupSize": [1, 1],
}

pixelate_by_thread = {
    "shader": """
// Shader where each thread independently processes its own region of pixels.
// Using workgroup_size(16,16) means we launch 16x16=256 threads per workgroup.
// Each thread calculates its own unique region based on its global_id.
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> uniforms: vec4<f32>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id : vec3<u32>,  // Unique ID for this thread across all workgroups
    @builtin(workgroup_id) workgroup_id : vec3<u32>        // ID of this thread's workgroup
) {
    let dims = textureDimensions(inputTex);

    // Block size controlled by the slider
    let blockSize = max(2.0, uniforms.w);

    // First calculate how many blocks we need total to cover the image
    let numBlocksX = ceil(f32(dims.x) / blockSize);
    let numBlocksY = ceil(f32(dims.y) / blockSize);

    // Then calculate the actual size each block should be to ensure even coverage
    // This prevents uneven blocks at the edges
    let normalizedBlockSizeX = f32(dims.x) / numBlocksX;
    let normalizedBlockSizeY = f32(dims.y) / numBlocksY;

    // Each thread uses its global_id to determine which block it owns
    // global_id ensures each thread processes a unique region with no overlap
    let blockX = u32(round(f32(global_id.x) * normalizedBlockSizeX));
    let blockY = u32(round(f32(global_id.y) * normalizedBlockSizeY));

    // Skip if this thread's block is outside the texture
    if (blockX >= dims.x || blockY >= dims.y) {
        return;
    }

    // Calculate where this thread's block ends by looking at where the next thread would start
    let nextBlockX = u32(round(f32(global_id.x + 1u) * normalizedBlockSizeX));
    let nextBlockY = u32(round(f32(global_id.y + 1u) * normalizedBlockSizeY));
    let blockEndX = min(nextBlockX, dims.x);
    let blockEndY = min(nextBlockY, dims.y);

    // Each thread calculates average color for its own block of pixels
    var sum = vec4<f32>(0.0);
    var count = 0u;

    // Sum up all pixels in this thread's block
    for (var y = blockY; y < blockEndY; y++) {
        for (var x = blockX; x < blockEndX; x++) {
            sum += textureLoad(inputTex, vec2<i32>(i32(x), i32(y)), 0);
            count++;
        }
    }

    // Calculate the average color for this thread's block
    let avgColor = sum / f32(count);

    // Check if this thread is at the top-left of its workgroup (every 16th thread)
    let isWorkgroupTopLeft = (global_id.x % 16u == 0u && global_id.y % 16u == 0u);

    // Each thread writes its averaged color to its own block of pixels
    for (var y = blockY; y < blockEndY; y++) {
        for (var x = blockX; x < blockEndX; x++) {
            if (isWorkgroupTopLeft) {
                // Draw 4x4 red square at workgroup boundaries
                if (x < blockX + 4u && y < blockY + 4u) {
                    textureStore(outputTex, vec2<i32>(i32(x), i32(y)), vec4<f32>(1.0, 0.0, 0.0, 1.0));
                } else {
                    textureStore(outputTex, vec2<i32>(i32(x), i32(y)), avgColor);
                }
            } else if (x == blockX && y == blockY) {
                // Draw 2x2 red square for individual thread block corners
                if (x < blockX + 2u && y < blockY + 2u) {
                    textureStore(outputTex, vec2<i32>(i32(x), i32(y)), vec4<f32>(1.0, 0.0, 0.0, 1.0));
                } else {
                    textureStore(outputTex, vec2<i32>(i32(x), i32(y)), avgColor);
                }
            } else {
                textureStore(outputTex, vec2<i32>(i32(x), i32(y)), avgColor);
            }
        }
    }
}
""",
    "workgroupSize": [16, 16],
}

(
    Plot.Import("path:notebooks/webgpu_compute.js", refer_all=True)
    | Plot.Slider("intendedPixelBlockSize", init=8, range=[2, 200])
    | Plot.initialState(
        {
            "width": 640,
            "height": 480,
            "tint": [1.0, 1.0, 1.0],
            "pixelBlockSize": js(
                """
                             const w = $state.width;
                             const numBlocks = Math.ceil(w/$state.intendedPixelBlockSize);
                             return w/numBlocks;
                             """,
                expression=False,
            ),
            "activeTransforms": [],
            "shaders": {
                "pixelate": pixelate_by_workgroup,
                "invert": invert,
                "workgroup_threads_pixelate": pixelate_by_thread,
                "tint": tint,
            },
            "getTransforms": js("""() => {
                const transforms = [];
                if ($state.activeTransforms.includes('pixelate')) {
                    transforms.push($state.shaders.pixelate);
                }
                if ($state.activeTransforms.includes('workgroup_threads_pixelate')) {
                    transforms.push($state.shaders.workgroup_threads_pixelate);
                }
                if ($state.activeTransforms.includes('invert')) {
                    transforms.push($state.shaders.invert);
                }
                if ($state.activeTransforms.includes('tint')) {
                    transforms.push($state.shaders.tint);
                }
                return transforms;
            }"""),
        }
    )
    | [
        [
            "div",
            {"className": "flex space-x-4 mb-4"},
            [
                [
                    "button",
                    {
                        "onClick": js("""(e) => {
                            const transform = 'pixelate';
                            const idx = $state.activeTransforms.indexOf(transform);
                            if (idx === -1) {
                                $state.activeTransforms = [...$state.activeTransforms, transform];
                            } else {
                                $state.activeTransforms = $state.activeTransforms.filter(t => t !== transform);
                            }
                        }"""),
                        "className": "px-4 py-2 rounded border hover:bg-gray-200 data-[selected=true]:bg-blue-500 data-[selected=true]:text-white",
                        "data-selected": js(
                            "$state.activeTransforms.includes('pixelate')"
                        ),
                    },
                    "Pixelate (1 thread per workgroup)",
                ],
                [
                    "button",
                    {
                        "onClick": js("""(e) => {
                            const transform = 'workgroup_threads_pixelate';
                            const idx = $state.activeTransforms.indexOf(transform);
                            if (idx === -1) {
                                $state.activeTransforms = [...$state.activeTransforms, transform];
                            } else {
                                $state.activeTransforms = $state.activeTransforms.filter(t => t !== transform);
                            }
                        }"""),
                        "className": "px-4 py-2 rounded border hover:bg-gray-200 data-[selected=true]:bg-blue-500 data-[selected=true]:text-white",
                        "data-selected": js(
                            "$state.activeTransforms.includes('workgroup_threads_pixelate')"
                        ),
                    },
                    "Pixelate (16x16 threads per workgroup)",
                ],
                [
                    "button",
                    {
                        "onClick": js("""(e) => {
                            const transform = 'invert';
                            const idx = $state.activeTransforms.indexOf(transform);
                            if (idx === -1) {
                                $state.activeTransforms = [...$state.activeTransforms, transform];
                            } else {
                                $state.activeTransforms = $state.activeTransforms.filter(t => t !== transform);
                            }
                        }"""),
                        "className": "px-4 py-2 rounded border hover:bg-gray-200 data-[selected=true]:bg-blue-500 data-[selected=true]:text-white",
                        "data-selected": js(
                            "$state.activeTransforms.includes('invert')"
                        ),
                    },
                    "Invert",
                ],
                [
                    "button",
                    {
                        "onClick": js("""(e) => {
                            const transform = 'tint';
                            const idx = $state.activeTransforms.indexOf(transform);
                            if (idx === -1) {
                                $state.activeTransforms = [...$state.activeTransforms, transform];
                            } else {
                                $state.activeTransforms = $state.activeTransforms.filter(t => t !== transform);
                            }
                        }"""),
                        "className": "px-4 py-2 rounded border hover:bg-gray-200 data-[selected=true]:bg-blue-500 data-[selected=true]:text-white",
                        "data-selected": js("$state.activeTransforms.includes('tint')"),
                    },
                    "Tint",
                ],
            ],
        ],
        [
            js("WebGPUVideoView"),
            {
                "transforms": js("$state.getTransforms()"),
                "showSourceVideo": True,
                "uniforms": js(
                    "[$state.tint[0], $state.tint[1], $state.tint[2], $state.pixelBlockSize]"
                ),
                "width": js("$state.width"),
                "height": js("$state.height"),
            },
        ],
        [
            js("colorScrubber"),
            {
                "value": js("$state.tint || [0,0,0]"),
                "onInput": js("(e) => { $state.tint = e.target.value; }"),
            },
        ],
    ]
).save_html("notebooks/webgpu.html")
