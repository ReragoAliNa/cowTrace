import os
import numpy as np
from typing import Tuple, Dict, Any, Union, Optional

# Flags to track library availability
TENSORRT_AVAILABLE = False
PYCUDA_AVAILABLE = False

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    pass

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    PYCUDA_AVAILABLE = True
except (ImportError, Exception):
    pass


class LSNetEngine:
    """
    LSNet Inference Engine utilizing NVIDIA TensorRT for FP16 optimization
    and PyCUDA for asynchronous memory allocation and execution on CUDA streams.
    
    Provides a CPU/Simulated fallback mode if TensorRT or PyCUDA are not installed,
    ensuring code remains runnable in CPU environments.
    """
    def __init__(
        self,
        model_path: str,
        engine_path: Optional[str] = None,
        force_rebuild: bool = False,
        use_fp16: bool = True
    ):
        self.model_path = model_path
        # If no engine path is specified, default to replacing .onnx with .engine
        self.engine_path = engine_path or (model_path.replace(".onnx", ".engine") if ".onnx" in model_path else model_path + ".engine")
        self.use_fp16 = use_fp16
        
        self.logger = None
        self.engine = None
        self.context = None
        self.stream = None
        
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        self.is_pt = self.model_path.endswith(".pt")
        self.model = None
        
        if self.is_pt:
            if os.path.exists(self.model_path):
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                print(f"Loaded PyTorch YOLO model: {self.model_path}")
            else:
                print(f"[WARNING] PyTorch YOLO model file not found at: {self.model_path}. Running in fallback simulation mode.")
        elif TENSORRT_AVAILABLE and PYCUDA_AVAILABLE:
            self.logger = trt.Logger(trt.Logger.WARNING)
            self._init_tensorrt(force_rebuild)
        else:
            print("[WARNING] TensorRT or PyCUDA is not installed. LSNetEngine is running in CPU/Fallback simulation mode.")

    def _init_tensorrt(self, force_rebuild: bool):
        """Builds or loads the TensorRT engine, then allocates GPU buffers."""
        # 1. Build engine if it doesn't exist or force_rebuild is set
        if force_rebuild or not os.path.exists(self.engine_path):
            print(f"Building TensorRT engine from ONNX model: {self.model_path}")
            self._build_engine_from_onnx()
        else:
            print(f"Loading serialized TensorRT engine: {self.engine_path}")
            self._load_engine()
            
        # 2. Create Execution Context
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        # 3. Allocate device and host buffers
        self._allocate_buffers()

    def _build_engine_from_onnx(self):
        """Parses ONNX model and builds serialized TensorRT Engine with FP16 options."""
        # Setup builders and parser
        builder = trt.Builder(self.logger)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, self.logger)
        
        # Read ONNX file
        with open(self.model_path, "rb") as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(f"Parser Error: {parser.get_error(error)}")
                raise RuntimeError("Failed to parse ONNX model.")
                
        config = builder.create_builder_config()
        
        # Enable FP16 if supported and requested
        if self.use_fp16:
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print("FP16 Mode enabled successfully.")
            else:
                print("FP16 Mode requested but not supported on this platform. Falling back to FP32.")
                
        # Build and serialize
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine.")
            
        # Save engine file
        with open(self.engine_path, "wb") as f:
            f.write(serialized_engine)
            
        # Deserialize engine
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)

    def _load_engine(self):
        """Loads and deserializes an existing TensorRT engine file."""
        runtime = trt.Runtime(self.logger)
        with open(self.engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
    def _allocate_buffers(self):
        """Allocates host and device memory buffers dynamically based on bindings."""
        for i in range(self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            volume = trt.volume(shape)
            
            # Host page-locked memory
            host_mem = cuda.pagelocked_empty(volume, dtype)
            # Device CUDA memory
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # Append binding index
            self.bindings.append(int(device_mem))
            
            binding_info = {
                "name": binding_name,
                "shape": shape,
                "dtype": dtype,
                "host": host_mem,
                "device": device_mem
            }
            
            if self.engine.binding_is_input(i):
                self.inputs.append(binding_info)
            else:
                self.outputs.append(binding_info)

    def infer(self, preprocessed_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Executes forward propagation on PyTorch (.pt) model or TensorRT engine (.engine).
        
        Args:
            preprocessed_image: Preprocessed BGR image of shape (640, 640, 3) (uint8).
            
        Returns:
            Tuple of:
                - Bboxes: Numpy matrix of detected boxes [x1, y1, x2, y2, confidence, class].
                - Masks: Numpy matrix containing instance masks.
                - Keypoints: Numpy matrix containing pose estimation coordinates.
        """
        if self.is_pt and self.model is not None:
            # 1. Run inference using PyTorch YOLO with NMS thresholds
            results = self.model(preprocessed_image, verbose=False, conf=0.25, iou=0.45)
            result = results[0]
            
            # 2. Extract Bboxes: [x1, y1, x2, y2, score, class]
            if result.boxes is not None and len(result.boxes) > 0:
                bboxes = result.boxes.data.cpu().numpy()  # [N, 6]
            else:
                bboxes = np.empty((0, 6), dtype=np.float32)
                
            # 3. Extract Masks: [N, H, W] (returns actual binary masks if it's a Seg model, otherwise zeros)
            if result.masks is not None and len(result.masks) > 0:
                masks = result.masks.data.cpu().numpy()
            else:
                masks = np.zeros((len(bboxes), 28, 28), dtype=np.float32)
                
            # 4. Extract Keypoints: [N, K, 3] (returns actual keypoints if it's a Pose model, otherwise mock coords)
            if result.keypoints is not None and len(result.keypoints) > 0:
                kpts = result.keypoints.data.cpu().numpy()
            else:
                kpts = np.zeros((len(bboxes), 17, 3), dtype=np.float32)
                for i in range(len(bboxes)):
                    cx = (bboxes[i, 0] + bboxes[i, 2]) / 2.0
                    cy = (bboxes[i, 1] + bboxes[i, 3]) / 2.0
                    for k in range(17):
                        kpts[i, k, 0] = cx
                        kpts[i, k, 1] = cy
                        kpts[i, k, 2] = 1.0  # mock visibility
                        
            return bboxes, masks, kpts

        if not (TENSORRT_AVAILABLE and PYCUDA_AVAILABLE):
            return self._infer_fallback(preprocessed_image)
            
        # --- Preprocessing inside the engine ---
        # Convert BGR (OpenCV) to RGB, normalize [0, 1], transpose to (1, 3, 640, 640)
        img = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0) # Shape: (1, 3, 640, 640)
        
        # Copy input data to page-locked host buffer
        input_info = self.inputs[0]
        np.copyto(input_info["host"], img.ravel())
        
        # --- Asynchronous memory transfers and execution ---
        # Host to Device transfer
        cuda.memcpy_htod_async(input_info["device"], input_info["host"], self.stream)
        
        # Execute model asynchronously on CUDA stream
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Device to Host transfer
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
            
        # Synchronize CPU execution with stream completion
        self.stream.synchronize()
        
        # --- Reshape outputs to original dimensions ---
        # Outputs contain: bboxes, masks, keypoints
        # Retrieve hosts
        bboxes = self.outputs[0]["host"].reshape(self.outputs[0]["shape"])
        masks = self.outputs[1]["host"].reshape(self.outputs[1]["shape"])
        kpts = self.outputs[2]["host"].reshape(self.outputs[2]["shape"])
        
        return bboxes, masks, kpts

    def _infer_fallback(self, preprocessed_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """CPU Fallback simulation mode. Generates simulated predictions for testing."""
        # Simulated multi-task prediction matrix structures
        # Assume 3 detections
        # Bbox shape: (3, 6) -> [x1, y1, x2, y2, score, class_id]
        bboxes = np.array([
            [50, 200, 140, 310, 0.92, 0],   # cow 1
            [220, 350, 340, 420, 0.88, 0],  # cow 2
            [80, 120, 160, 225, 0.90, 0]     # cow 3
        ], dtype=np.float32)
        
        # Mask shape: (3, 28, 28) -> Instance Segmentation mask patches
        masks = np.random.uniform(0.0, 1.0, (3, 28, 28)).astype(np.float32)
        
        # Keypoints shape: (3, 17, 3) -> 17 Keypoints (x, y, visibility)
        kpts = np.zeros((3, 17, 3), dtype=np.float32)
        for i in range(3):
            # Generate simulated keypoint coords around centroid of bbox
            cx = (bboxes[i, 0] + bboxes[i, 2]) / 2.0
            cy = (bboxes[i, 1] + bboxes[i, 3]) / 2.0
            for k in range(17):
                kpts[i, k, 0] = cx + np.random.uniform(-30, 30)
                kpts[i, k, 1] = cy + np.random.uniform(-40, 40)
                kpts[i, k, 2] = 2.0 # visible and labeled
                
        return bboxes, masks, kpts
