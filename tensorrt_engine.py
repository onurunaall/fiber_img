import argparse
from pathlib import Path

import tensorrt as trt
import torch


TORCH_DTYPES: dict[trt.DataType, torch.dtype] = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.BF16: torch.bfloat16,
}


def build_engine(onnx_path: Path, engine_path: Path, max_batch_size: int) -> None:
    """Builds a TensorRT engine for batch sizes 1 to max_batch_size.

    TensorRT 11 networks are always strongly typed: every layer runs in the data type that
    the ONNX graph uses. The precision of the engine is therefore chosen at ONNX export.
    The engine only runs on the GPU model and TensorRT version it was built with.
    """
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(str(onnx_path)):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError(f"TensorRT could not parse {onnx_path}:\n" + "\n".join(errors))

    if network.num_inputs != 1:
        raise ValueError(f"Expected a network with one input, found {network.num_inputs}.")

    network_input = network.get_input(0)
    input_shape = tuple(network_input.shape)
    if len(input_shape) != 4:
        raise ValueError(f"Expected a 4D input (batch, channels, height, width), found {input_shape}.")

    _, channels, height, width = input_shape
    smallest_shape = (1, channels, height, width)
    largest_shape = (max_batch_size, channels, height, width)

    profile = builder.create_optimization_profile()
    profile.set_shape(network_input.name, smallest_shape, largest_shape, largest_shape)

    config = builder.create_builder_config()
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT could not build the engine. See the TensorRT messages above.")

    engine_path.write_bytes(serialized_engine)


def find_single_tensor_name(engine: trt.ICudaEngine, mode: trt.TensorIOMode) -> str:
    names: list[str] = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) == mode:
            names.append(name)

    if len(names) != 1:
        raise ValueError(f"Expected exactly one {mode.name} tensor in the engine, found {len(names)}.")
    return names[0]


class TensorRTModel:
    """Runs a TensorRT engine on PyTorch CUDA tensors, like calling an nn.Module in eval mode."""

    def __init__(self, engine_path: Path, device: torch.device) -> None:
        if device.type != "cuda":
            raise ValueError(f"TensorRT needs a CUDA device, got '{device}'.")

        self.device = device
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)

        with torch.cuda.device(device):
            engine = self._runtime.deserialize_cuda_engine(engine_path.read_bytes())
            if engine is None:
                raise RuntimeError(f"TensorRT could not load {engine_path}. See the TensorRT messages above.")
            self._context = engine.create_execution_context()

        self._engine = engine
        self._input_name = find_single_tensor_name(engine, trt.TensorIOMode.INPUT)
        self._output_name = find_single_tensor_name(engine, trt.TensorIOMode.OUTPUT)
        self._input_dtype = TORCH_DTYPES[engine.get_tensor_dtype(self._input_name)]
        self._output_dtype = TORCH_DTYPES[engine.get_tensor_dtype(self._output_name)]

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.to(device=self.device, dtype=self._input_dtype).contiguous()

        with torch.cuda.device(self.device):
            if not self._context.set_input_shape(self._input_name, tuple(inputs.shape)):
                raise ValueError(f"Input shape {tuple(inputs.shape)} is outside the range the engine was built for.")

            output_shape = tuple(self._context.get_tensor_shape(self._output_name))
            outputs = torch.empty(output_shape, dtype=self._output_dtype, device=self.device)

            self._context.set_tensor_address(self._input_name, inputs.data_ptr())
            self._context.set_tensor_address(self._output_name, outputs.data_ptr())

            # Running on PyTorch's current stream orders the engine correctly with the surrounding PyTorch operations.
            stream = torch.cuda.current_stream(self.device)
            if not self._context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("TensorRT inference failed. See the TensorRT messages above.")

        return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a TensorRT engine from an ONNX file.")
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--engine", required=True, type=Path, help="path of the .engine file to write")
    parser.add_argument("--max_batch_size", default=4, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_engine(args.onnx, args.engine, args.max_batch_size)
    print(f"Saved {args.engine}")


if __name__ == "__main__":
    main()
