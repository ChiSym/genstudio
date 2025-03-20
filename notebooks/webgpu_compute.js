// ----- "webgpuVideoCompute.js" -----
//
// A React component that processes webcam video through a WebGPU compute shader.
// The user provides the WGSL compute shader code as a string prop.
//
// Props:
// - transform: Object containing:
//   - shader: WGSL compute shader code as a string
//   - workgroupSize: Array with two elements [x, y] for compute shader workgroup size (default: [8, 8])
//   - dispatchScale: Multiplier for workgroup dispatch calculation (default: 1)
//   - customDispatch: Array with two elements [x, y] for direct workgroup count control (default: null)
// - width: Canvas width (default: 640)
// - height: Canvas height (default: 480)
// - showSourceVideo: Whether to show the source video (default: false)
// - uniforms: An object of uniforms to be copied into the WebGPU context and available in the compute shader
// - debug: Enable debug logging (default: false)
//
// The compute shader should:
// - Use @group(0) @binding(0) for the input texture (texture_2d<f32>)
// - Use @group(0) @binding(1) for the output texture (texture_storage_2d<rgba8unorm, write>)
// - Use @group(0) @binding(2) for the uniform buffer containing the uniforms
// - Have a main() function with @compute @workgroup_size(x, y) decorator matching workgroupSize prop
// - Take a @builtin(global_invocation_id) parameter to get pixel coordinates
// - Check texture bounds before processing pixels
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

  // Create arrays to store pipelines and bind groups
  const computePipelines = [];
  const computeBindGroups = [];
  const intermediateTextures = [];

  // Create intermediate textures for each transform except the last one
  for (let i = 0; i < transforms.length - 1; i++) {
    const texture = device.createTexture({
      size: [state.width, state.height],
      format: "rgba8unorm",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
    });
    intermediateTextures.push(texture);
  }

  // Create pipeline and bind group for each transform
  transforms.forEach((transform, i) => {
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

    // For each transform, input is either inputTexture (first transform) or previous intermediate texture
    // Output is either outputTexture (last transform) or next intermediate texture
    const inputView = i === 0 ?
      state.inputTexture.createView() :
      intermediateTextures[i-1].createView();

    const outputView = i === transforms.length - 1 ?
      state.outputTexture.createView() :
      intermediateTextures[i].createView();

    const computeBindGroup = device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: inputView },
        { binding: 1, resource: outputView },
        { binding: 2, resource: { buffer: state.uniformBuffer } },
      ],
    });

    computePipelines.push(computePipeline);
    computeBindGroups.push(computeBindGroup);
  });

  // Store in state
  state.computePipelines = computePipelines;
  state.computeBindGroups = computeBindGroups;
  state.intermediateTextures = intermediateTextures;
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

  // Apply each transform in sequence
  transforms.forEach((transform, i) => {
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

export const WebGPUVideoView = ({
  transform = [],  // Now accepts an array of transforms
  width = 640,
  height = 480,
  showSourceVideo = false,
  uniforms = {},
  debug = false
}) => {
  const canvasId = React.useId();
  const frameRef = React.useRef(null);

  // Convert single transform to array if needed
  const transforms = Array.isArray(transform) ? transform : [transform];
  transforms = transforms.map(t => transformWithDefaults(t));
  const transformsRef = React.useRef(transforms);
  transformsRef.current = transforms;

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
    computePipelines: [],  // Now an array
    computeBindGroups: [], // Now an array
    intermediateTextures: [], // New array for intermediate textures
    renderPipeline: null,
    renderBindGroup: null,
    uniformBuffer: null,
    width,
    height,
    debug,
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

  // Initialize WebGPU resources and start rendering
  React.useEffect(() => {
    setupWebGPUResources(webgpuRef.current, {
      width,
      height,
      canvasId,
      uniforms,
    })
      .then(() => {
        // After initial setup, create the compute pipelines from the provided shaders
        updateComputePipelines(webgpuRef.current, transforms);
        const animate = () => {
          frameRef.current = requestAnimationFrame(animate);
          renderFrame(webgpuRef.current, transformsRef.current, width, height);
        };
        animate();
      })
      .catch((error) => {
        console.error("Setup failed:", error);
      });

    return () => {
      if (frameRef.current) {
        cancelAnimationFrame(frameRef.current);
      }
      const { video, inputTexture, outputTexture, uniformBuffer, intermediateTextures } = webgpuRef.current;
      if (video?.srcObject) {
        video.srcObject.getTracks().forEach((track) => track.stop());
      }
      inputTexture?.destroy();
      outputTexture?.destroy();
      uniformBuffer?.destroy();
      intermediateTextures?.forEach(texture => texture?.destroy());
    };
  }, [width, height, canvasId]);

  // Update compute pipelines if transforms change without restarting everything
  React.useEffect(() => {
    if (webgpuRef.current.device) {
      // Clean up old intermediate textures
      webgpuRef.current.intermediateTextures?.forEach(texture => texture?.destroy());
      updateComputePipelines(webgpuRef.current, transforms);
    }
  }, [transforms]);

  // Update uniforms on change using helper function
  React.useEffect(() => {
    updateUniformBuffer(webgpuRef.current.device, webgpuRef.current.uniformBuffer, uniforms);
  }, [uniforms]);

  return html(["canvas", { id: canvasId, width, height }]);
};
