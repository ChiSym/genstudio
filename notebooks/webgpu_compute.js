// ----- "webgpuVideoCompute.js" -----
//
// A React component that processes webcam video through one or more WebGPU compute shaders.
// The user provides an array of WGSL compute shader configurations as props.
//
// Props:
// - transforms: Array of transform objects, each containing:
//   - shader: WGSL compute shader code as a string
//   - workgroupSize: Array with two elements [x, y] for compute shader workgroup size (default: [8, 8])
//   - dispatchScale: Multiplier for workgroup dispatch calculation (default: 1)
//   - customDispatch: Array with two elements [x, y] for direct workgroup count control (default: null)
// - width: Canvas width (default: 640)
// - height: Canvas height (default: 480)
// - showSourceVideo: Whether to show the source video (default: false)
// - uniforms: An object of uniforms to be copied into the WebGPU context and available in all compute shaders
// - debug: Enable debug logging (default: false)
//
// Each compute shader should:
// - Use @group(0) @binding(0) for the input texture (texture_2d<f32>)
// - Use @group(0) @binding(1) for the output texture (texture_storage_2d<rgba8unorm, write>)
// - Use @group(0) @binding(2) for the uniform buffer containing the uniforms
// - Have a main() function with @compute @workgroup_size(x, y) decorator matching workgroupSize prop
// - Take a @builtin(global_invocation_id) parameter to get pixel coordinates
// - Check texture bounds before processing pixels
//
// Transforms are applied sequentially, with the output of each transform feeding into the next.
// The first transform receives the webcam input, and the final transform's output is displayed.
//
// Example compute shader:
//
// @group(0) @binding(0) var inputTex : texture_2d<f32>;
// @group(0) @binding(1) var outputTex : texture_storage_2d<rgba8unorm, write>;
// @group(0) @binding(2) var<uniform> uniforms: MyUniforms;
//
// @compute @workgroup_size(8, 8)
// fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
//   let dims = textureDimensions(inputTex);
//   if (gid.x >= dims.x || gid.y >= dims.y) { return; }
//
//   let srcColor = textureLoad(inputTex, vec2<i32>(gid.xy), 0);
//   // Process srcColor here...
//   textureStore(outputTex, vec2<i32>(gid.xy), outColor);
// }

const { html, React } = genstudio.api;
const { useState } = React;
export const colorScrubber = ({ value, onInput }) => {
  // Use internal state if no external value is provided (uncontrolled mode)
  const [internalColor, setInternalColor] = useState([1, 0, 0]);
  const currentColor = value !== undefined ? value : internalColor;

  // Helper: Convert HSL (with s:100%, l:50%) to RGB array
  const hslToRgb = (h) => {
    h = h / 360;
    const s = 1.0;
    const l = 0.5;
    const k = (n) => (n + h * 12) % 12;
    const a = s * Math.min(l, 1 - l);
    const f = (n) => l - a * Math.max(Math.min(k(n) - 3, 9 - k(n), 1), -1);
    return [f(0), f(8), f(4)];
  };

  // Helper: Convert RGB array to hue (0-360)
  const rgbToHue = (rgb) => {
    const [r, g, b] = rgb;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    let h;
    if (max === min) {
      h = 0;
    } else {
      const d = max - min;
      if (max === r) {
        h = (g - b) / d + (g < b ? 6 : 0);
      } else if (max === g) {
        h = (b - r) / d + 2;
      } else {
        h = (r - g) / d + 4;
      }
      h /= 6;
    }
    return Math.round(h * 360);
  };

  // Linear gradient for the slider background using RGB values
  const gradientBackground = `linear-gradient(to right, ${Array.from({ length: 12 }, (_, i) => {
    const [r, g, b] = hslToRgb(i * 30);
    return `rgb(${r * 255}, ${g * 255}, ${b * 255})`;
  }).join(', ')
    })`;

  const handleColorChange = (e) => {
    if (e.target.type === 'range') {
      const newHue = parseInt(e.target.value, 10);
      const newColor = hslToRgb(newHue);
      if (onInput) {
        onInput({ target: { value: newColor } });
      }
      if (value === undefined) {
        setInternalColor(newColor);
      }
    }
  };

  return html(
    [
      "div.h-10.w-full.rounded-full.mb-4.overflow-hidden",
      { style: { background: gradientBackground } },
      [
        "input",
        {
          type: "range",
          min: "0",
          max: "360",
          value: rgbToHue(currentColor),
          onChange: handleColorChange,
          className: `
        w-full h-full appearance-none bg-transparent
        [&::-webkit-slider-thumb]:(border appearance-none rounded-full bg-white w-[20px] h-[20px])`,
        },
      ],
    ]
  );
};

function setupVideoCanvas(width, height, showSourceVideo) {
  const videoCanvas = document.createElement("canvas");
  videoCanvas.width = width;
  videoCanvas.height = height;
  const videoCtx = videoCanvas.getContext("2d");

  if (showSourceVideo) {
    Object.assign(videoCanvas.style, {
      position: "fixed",
      bottom: "10px",
      right: "10px",
      border: "1px solid red",
      width: "160px",
      height: "120px",
      zIndex: "1000",
    });
    document.body.appendChild(videoCanvas);
  }

  return {
    videoCanvas,
    videoCtx,
    cleanup: () => {
      if (showSourceVideo) {
        document.body.removeChild(videoCanvas);
      }
    },
  };
}

async function setupWebcam(state, width, height) {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: width },
        height: { ideal: height },
      },
    });

    const video = document.createElement("video");
    Object.assign(video, {
      srcObject: stream,
      width,
      height,
      autoplay: true,
      playsInline: true,
      muted: true,
    });

    await new Promise((resolve) => {
      video.onloadedmetadata = () => video.play().then(resolve);
    });

    state.video = video;
  } catch (error) {
    console.error("Webcam setup failed:", error);
    throw error;
  }
}

async function initWebGPU(state, canvasId) {
  if (!navigator.gpu) {
    throw new Error("WebGPU not supported");
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("Failed to get GPU adapter");
  }
  const device = await adapter.requestDevice({validationEnabled: true});
  const canvas = document.getElementById(canvasId);
  const context = canvas.getContext("webgpu");

  if (!context) {
    throw new Error("Failed to get WebGPU context");
  }

  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format,
    alphaMode: "premultiplied",
  });

  state.device = device;
  state.context = context;
  state.renderFormat = format;

  return { device, context, format };
}

// Helper function to create uniform buffer from uniforms object
function createUniformBuffer(device, uniforms) {
  const uniformKeys = Object.keys(uniforms).sort();
  const uniformArray = uniformKeys.map((key) => uniforms[key]);
  const uniformData = new Float32Array(
    uniformArray.length > 0 ? uniformArray : [0]
  );
  const uniformBuffer = device.createBuffer({
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Float32Array(uniformBuffer.getMappedRange()).set(uniformData);
  uniformBuffer.unmap();
  return uniformBuffer;
}

// Helper function to update uniform buffer with new values
function updateUniformBuffer(device, uniformBuffer, uniforms) {
  if (!device || !uniformBuffer) return;
  const uniformKeys = Object.keys(uniforms).sort();
  const uniformArray = uniformKeys.map((key) => uniforms[key]);
  const uniformData = new Float32Array(
    uniformArray.length > 0 ? uniformArray : [0]
  );
  device.queue.writeBuffer(
    uniformBuffer,
    0,
    uniformData.buffer,
    uniformData.byteOffset,
    uniformData.byteLength
  );
}

async function setupWebGPUResources(state, { width, height, canvasId, uniforms }) {
  const { device, context, format } = await initWebGPU(state, canvasId);
  await setupWebcam(state, width, height);

  const usage = GPUTextureUsage;
  const textureFormat = "rgba8unorm";

  const inputTexture = device.createTexture({
    size: [width, height],
    format: textureFormat,
    usage:
      usage.COPY_SRC |
      usage.COPY_DST |
      usage.TEXTURE_BINDING |
      usage.RENDER_ATTACHMENT,
  });

  const outputTexture = device.createTexture({
    size: [width, height],
    format: textureFormat,
    usage:
      usage.STORAGE_BINDING |
      usage.TEXTURE_BINDING |
      usage.COPY_DST |
      usage.RENDER_ATTACHMENT,
  });

  // Create render pipeline and resources
  const vertexShaderCode = /* wgsl */ `
    struct VertexOutput {
      @builtin(position) position: vec4<f32>,
      @location(0) texCoord: vec2<f32>,
    };

    @vertex
    fn vsMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
      var output: VertexOutput;
      var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0)
      );
      var texCoords = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(1.0, 0.0)
      );
      output.position = vec4<f32>(positions[vertexIndex], 0.0, 1.0);
      output.texCoord = texCoords[vertexIndex];
      return output;
    }
  `;

  const fragmentShaderCode = /* wgsl */ `
    @group(0) @binding(0) var myTex: texture_2d<f32>;
    @group(0) @binding(1) var mySampler: sampler;

    @fragment
    fn fsMain(@location(0) texCoord: vec2<f32>) -> @location(0) vec4<f32> {
      return textureSample(myTex, mySampler, texCoord);
    }
  `;

  const renderModule = device.createShaderModule({
    code: vertexShaderCode + fragmentShaderCode,
  });

  const renderPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: renderModule,
      entryPoint: "vsMain",
    },
    fragment: {
      module: renderModule,
      entryPoint: "fsMain",
      targets: [{ format }],
    },
    primitive: {
      topology: "triangle-list",
    },
  });

  const sampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
    addressModeU: "clamp-to-edge",
    addressModeV: "clamp-to-edge",
  });

  const renderBindGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: outputTexture.createView() },
      { binding: 1, resource: sampler },
    ],
  });

  // Note: Compute pipeline creation is deferred, allowing it to be swapped later
  state.inputTexture = inputTexture;
  state.outputTexture = outputTexture;
  state.sampler = sampler;
  state.renderPipeline = renderPipeline;
  state.renderBindGroup = renderBindGroup;
  state.uniformBuffer = createUniformBuffer(device, uniforms);
}

function transformWithDefaults(transform = {}) {
  return {
    shader: transform.shader ?? '',
    workgroupSize: transform.workgroupSize ?? [16, 16],
    dispatchScale: transform.dispatchScale ?? 1,
    customDispatch: transform.customDispatch ?? null
  };
}

function updateComputePipelines(state, transforms) {
  const device = state.device;
  if (!device) return;

  // Clean up existing resources that won't be reused
  if (state.intermediateTextures?.length > transforms.length - 1) {
    // Destroy extra intermediate textures
    for (let i = transforms.length - 1; i < state.intermediateTextures.length; i++) {
      state.intermediateTextures[i]?.destroy();
    }
  }

  // Create arrays to store pipelines and bind groups
  const computePipelines = [];
  const computeBindGroups = [];
  const intermediateTextures = state.intermediateTextures || [];

  // Resize intermediate textures array if needed
  intermediateTextures.length = Math.max(0, transforms.length - 1);

  // Create or reuse intermediate textures for each transform except the last one
  for (let i = 0; i < transforms.length - 1; i++) {
    if (!intermediateTextures[i]) {
      const texture = device.createTexture({
        size: [state.width, state.height],
        format: "rgba8unorm",
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
      });
      intermediateTextures[i] = texture;
    }
  }

  // Create pipeline and bind group for each transform
  transforms.forEach((transform, i) => {
    // For each transform, input is either inputTexture (first transform) or previous intermediate texture
    // Output is either outputTexture (last transform) or next intermediate texture
    const inputView = i === 0 ?
      state.inputTexture.createView() :
      intermediateTextures[i-1].createView();

    const outputView = i === transforms.length - 1 ?
      state.outputTexture.createView() :
      intermediateTextures[i].createView();

    // Use the helper function to create pipeline and bind group
    const { pipeline, bindGroup } = createTransformPipeline(state, transform, inputView, outputView);

    computePipelines.push(pipeline);
    computeBindGroups.push(bindGroup);
  });

  // Store in state
  state.computePipelines = computePipelines;
  state.computeBindGroups = computeBindGroups;
  state.intermediateTextures = intermediateTextures;

  if (state.debug) {
    console.log(`[DEBUG] Updated compute pipelines: ${transforms.length} transforms with ${intermediateTextures.length} intermediate textures`);
  }
}

async function renderFrame(state, transforms, width, height) {
  const {
    video,
    videoCanvas,
    videoCtx,
    device,
    context,
    inputTexture,
    renderPipeline,
    renderBindGroup,
    computePipelines,
    computeBindGroups,
  } = state;

  if (!video || video.readyState < 3 || video.paused) return;

  // Draw video frame to canvas
  videoCtx.drawImage(video, 0, 0, width, height);

  try {
    const imageBitmap = await createImageBitmap(videoCanvas);
    device.queue.copyExternalImageToTexture(
      { source: imageBitmap },
      { texture: inputTexture },
      [width, height]
    );
  } catch (error) {
    const imageData = videoCtx.getImageData(0, 0, width, height);
    device.queue.writeTexture(
      { texture: inputTexture },
      imageData.data,
      { bytesPerRow: width * 4 },
      { width, height, depthOrArrayLayers: 1 }
    );
  }

  // Encode and submit commands
  const commandEncoder = device.createCommandEncoder();

  // If no transforms, render input directly
  if (!transforms.length || !computePipelines.length) {
    // Create a special render bind group that uses the input texture instead
    const directRenderBindGroup = device.createBindGroup({
      layout: renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: inputTexture.createView() },
        { binding: 1, resource: state.sampler },
      ],
    });

    const view = context.getCurrentTexture().createView();
    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view,
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    renderPass.setPipeline(renderPipeline);
    renderPass.setBindGroup(0, directRenderBindGroup);
    renderPass.draw(6, 1, 0, 0);
    renderPass.end();

    device.queue.submit([commandEncoder.finish()]);
    return;
  }

  // Apply each transform in sequence
  transforms.forEach((transform, i) => {
    // Skip if pipeline is missing
    if (!computePipelines[i] || !computeBindGroups[i]) return;

    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(computePipelines[i]);
    computePass.setBindGroup(0, computeBindGroups[i]);

    const [wgX, wgY] = transform.customDispatch || [
      Math.ceil(width / (transform.workgroupSize[0] * transform.dispatchScale)),
      Math.ceil(height / (transform.workgroupSize[1] * transform.dispatchScale)),
    ];

    if (state.debug) {
      console.log(`[DEBUG] Transform ${i}: Dispatching workgroups: (${wgX}, ${wgY})`);
    }

    computePass.dispatchWorkgroups(wgX, wgY);
    computePass.end();
  });

  const view = context.getCurrentTexture().createView();
  const renderPass = commandEncoder.beginRenderPass({
    colorAttachments: [
      {
        view,
        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  });
  renderPass.setPipeline(renderPipeline);
  renderPass.setBindGroup(0, renderBindGroup);
  renderPass.draw(6, 1, 0, 0);
  renderPass.end();

  device.queue.submit([commandEncoder.finish()]);
}

// Helper function for deep comparison of transforms
function transformsDeepEqual(transformsA, transformsB) {
  if (transformsA === transformsB) return true;
  if (!transformsA || !transformsB) return false;
  if (transformsA.length !== transformsB.length) return false;

  return transformsA.every((transformA, index) => {
    const transformB = transformsB[index];

    // Compare shader code
    if (transformA.shader !== transformB.shader) return false;

    // Compare workgroup size
    if (!transformA.workgroupSize || !transformB.workgroupSize) return false;
    if (transformA.workgroupSize.length !== transformB.workgroupSize.length) return false;
    if (transformA.workgroupSize[0] !== transformB.workgroupSize[0] ||
        transformA.workgroupSize[1] !== transformB.workgroupSize[1]) return false;

    // Compare dispatch scale
    if (transformA.dispatchScale !== transformB.dispatchScale) return false;

    // Compare custom dispatch
    if (transformA.customDispatch === null && transformB.customDispatch === null) return true;
    if (!transformA.customDispatch || !transformB.customDispatch) return false;
    if (transformA.customDispatch.length !== transformB.customDispatch.length) return false;
    return transformA.customDispatch[0] === transformB.customDispatch[0] &&
           transformA.customDispatch[1] === transformB.customDispatch[1];
  });
}

// Creates a compute pipeline and bind group for a single transform
function createTransformPipeline(state, transform, inputView, outputView) {
  const { device } = state;
  if (!device) return null;

  const workgroupSize = transform.workgroupSize || [16, 16];

  const computeModule = device.createShaderModule({
    code: transform.shader,
  });

  const computePipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: computeModule,
      entryPoint: "main",
      workgroupSize: { x: workgroupSize[0], y: workgroupSize[1] },
    },
  });

  const computeBindGroup = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: inputView },
      { binding: 1, resource: outputView },
      { binding: 2, resource: { buffer: state.uniformBuffer } },
    ],
  });

  return { pipeline: computePipeline, bindGroup: computeBindGroup };
}

export const WebGPUVideoView = ({
  transforms = [],  // Now accepts an array of transforms
  width = 640,
  height = 480,
  showSourceVideo = false,
  uniforms = {},
  debug = false
}) => {
  const canvasId = React.useId();
  const frameRef = React.useRef(null);

  // Convert single transform to array if needed
  transforms = Array.isArray(transforms) ? transforms : [transforms];
  transforms = transforms.map(t => transformWithDefaults(t));

  // Use useRef and a dependency tracker to detect changes in transforms
  const prevTransformsRef = React.useRef([]);
  const transformsRef = React.useRef(transforms);

  // Store transform state including what needs updating
  const [transformState, setTransformState] = React.useState({
    transforms: transforms,
    needsUpdate: true,
    lastUpdateTime: Date.now() // Track when we last updated for performance logging
  });

  // Group state for WebGPU and video, including the debug flag
  const webgpuRef = React.useRef({
    video: null,
    videoCanvas: null,
    videoCtx: null,
    device: null,
    context: null,
    renderFormat: null,
    inputTexture: null,
    outputTexture: null,
    sampler: null,
    computePipelines: [],
    computeBindGroups: [],
    intermediateTextures: [],
    renderPipeline: null,
    renderBindGroup: null,
    uniformBuffer: null,
    width,
    height,
    debug,
    initialized: false
  });

  // Setup video canvas
  React.useEffect(() => {
    const { videoCanvas, videoCtx, cleanup } = setupVideoCanvas(
      width,
      height,
      showSourceVideo
    );
    webgpuRef.current.videoCanvas = videoCanvas;
    webgpuRef.current.videoCtx = videoCtx;
    return cleanup;
  }, [width, height, showSourceVideo]);

  // Check for deep changes in transforms
  React.useEffect(() => {
    const hasChanged = !transformsDeepEqual(prevTransformsRef.current, transforms);
    if (hasChanged) {
      if (debug) {
        console.log("[DEBUG] Detected changes in transforms configuration");
      }
      prevTransformsRef.current = transforms;
      transformsRef.current = transforms;
      setTransformState({
        transforms: transforms,
        needsUpdate: true,
        lastUpdateTime: Date.now()
      });
    }
  }, [transforms, debug]);

  // Initialize WebGPU resources and start rendering
  React.useEffect(() => {
    if (webgpuRef.current.initialized) {
      // Just update dimensions if already initialized
      webgpuRef.current.width = width;
      webgpuRef.current.height = height;
      return;
    }

    setupWebGPUResources(webgpuRef.current, {
      width,
      height,
      canvasId,
      uniforms,
    })
      .then(() => {
        // After initial setup, create the compute pipelines from the provided shaders
        updateComputePipelines(webgpuRef.current, transformsRef.current);
        webgpuRef.current.initialized = true;

        const animate = () => {
          frameRef.current = requestAnimationFrame(animate);
          // Always use the latest transforms from the ref
          const currentTransforms = transformsRef.current;
          renderFrame(webgpuRef.current, currentTransforms, width, height);
        };
        animate();
      })
      .catch((error) => {
        console.error("WebGPU setup failed:", error);
      });

    return () => {
      if (frameRef.current) {
        cancelAnimationFrame(frameRef.current);
      }

      // Clean up WebGPU resources
      const {
        video,
        inputTexture,
        outputTexture,
        uniformBuffer,
        intermediateTextures,
        computePipelines
      } = webgpuRef.current;

      if (video?.srcObject) {
        video.srcObject.getTracks().forEach((track) => track.stop());
      }

      // Destroy textures
      inputTexture?.destroy();
      outputTexture?.destroy();
      uniformBuffer?.destroy();

      // Clean up intermediate textures
      if (intermediateTextures?.length) {
        intermediateTextures.forEach(texture => texture?.destroy());
      }

      webgpuRef.current.initialized = false;
    };
  }, [canvasId]); // Only depend on canvasId since we handle width/height changes separately

  // Handle resize by recreating textures if needed
  React.useEffect(() => {
    const state = webgpuRef.current;

    // Skip if not initialized or dimensions haven't changed
    if (!state.initialized || (state.width === width && state.height === height)) {
      return;
    }

    // Update dimensions in state
    state.width = width;
    state.height = height;

    // Recreate textures with new dimensions
    // This would require more complex logic, but for now we'll trigger a full reinitialization
    setTransformState(prev => ({...prev, needsUpdate: true}));

  }, [width, height]);

  // Update compute pipelines only when transforms actually change (using deep comparison)
  React.useEffect(() => {
    if (webgpuRef.current.device && transformState.needsUpdate) {
      const startTime = performance.now();

      // Update compute pipelines with new transforms
      updateComputePipelines(webgpuRef.current, transformState.transforms);

      // Mark as updated
      setTransformState(prev => ({
        ...prev,
        needsUpdate: false,
        lastUpdateTime: Date.now()
      }));

      if (debug) {
        const duration = performance.now() - startTime;
        console.log(`[DEBUG] Transforms updated in ${duration.toFixed(2)}ms for ${transformState.transforms.length} transform(s)`);
      }
    }
  }, [transformState, debug]);

  // Update uniforms on change using helper function
  React.useEffect(() => {
    updateUniformBuffer(webgpuRef.current.device, webgpuRef.current.uniformBuffer, uniforms);
  }, [uniforms]);

  return html(["canvas", { id: canvasId, width, height }]);
};


// PROPOSAL - COMPUTE GRAPH
//
// A declarative specification for WebGPU compute pipelines that enables:
// - Complex multi-pass compute operations
// - Resource lifetime management
// - Automatic dependency resolution
// - Flexible buffer/texture configurations
//
// Example compute graph configuration:
// {
//   // Define all resources (buffers/textures) used in the graph
//   resources: {
//     // Texture for edge detection output
//     edges: {
//       type: 'texture',
//       format: 'rgba8unorm',
//       usage: ['storage', 'sampled'],
//       size: [width, height]
//     },
//
//     // Parameters buffer for edge detection
//     params: {
//       type: 'buffer',
//       format: 'float32',
//       size: 1024,  // Size in bytes
//       usage: ['storage', 'uniform'],
//       data: new Float32Array([...]) // Optional initial data
//     },
//
//     // Histogram buffer for analysis
//     hist: {
//       type: 'buffer',
//       format: 'uint32',
//       size: 256,
//       usage: ['storage'],
//       clear: true // Clear buffer each frame
//     }
//   },
//
//   // Define compute shader nodes and their connections
//   nodes: {
//     detect: {
//       shader: '...', // WGSL shader code
//       workgroups: [16,16],
//       inputs: {
//         src: 'source',    // Special 'source' refers to input texture
//         params: 'params'  // Reference to params buffer
//       },
//       outputs: {
//         edges: 'edges',   // Write to edges texture
//         hist: 'hist'      // Write to histogram buffer
//       },
//       uniforms: {        // Optional uniform values
//         threshold: 0.5,
//         kernelSize: 3
//       }
//     },
//
//     // Additional nodes can be added for multi-pass effects
//     blur: {
//       shader: '...',
//       workgroups: [8,8],
//       inputs: { src: 'edges' },
//       outputs: { dst: 'screen' }
//     }
//   },
//
//   // Special nodes for input/output
//   entry: 'source',   // Input texture
//   output: 'screen'   // Final output
// }
//
// Benefits:
// - Explicit resource definitions with clear lifetime management
// - Named inputs/outputs for better code organization
// - Flexible buffer/texture configurations
// - Clear data flow between shader passes
// - WebGPU-aligned specifications
// - Automatic resource cleanup
// - Built-in validation and error checking
// - Simplified multi-pass setup
